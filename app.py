from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from fuzzy_module import calcular_riesgo_difuso
import sqlite3
from datetime import datetime
#NUEVOS IMPORTS#
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)

# =================== FIREBASE - INICIO ===================
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "projectId": "riesgo-crediticio-1dbae"
})
db = firestore.client()
# ==================== FIREBASE - FIN =====================

def guardar_evaluacion_en_firebase(entrada, resultado_modelo, riesgo_difuso):
    """
    Guarda una evaluación de crédito en Firebase Firestore.
    """
    try:
        db.collection("evaluaciones_credito").add({
            "entrada": entrada,                  # lo que envió el usuario
            "resultado_modelo": resultado_modelo,  # por ejemplo probabilidad / clase
            "riesgo_difuso": riesgo_difuso,        # el valor que te da calcular_riesgo_difuso
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error guardando en Firebase: {e}")
# ==================== PRUEBA =====================


# Cargar modelo y scaler
modelo = joblib.load('models/Random_Forest_modelo_final.pkl')
scaler = joblib.load('models/scaler_datos.pkl')

# Mapeos para variables categóricas
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Region': {'North': 0, 'South': 1, 'Central': 2, 'North-East': 3},
    'credit_type': {'CIB': 0, 'EXP': 1, 'EQUI': 2},
    'age': {'<25': 6, '25-34': 0, '35-44': 1, '45-54': 2, '55-64': 3, '65-74': 4, '>74': 5},
    'credit_worthiness': {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1},
    'loan_purpose': {'Personal': 0, 'Hipoteca': 1, 'Auto': 2, 'Comercial': 3, 'Educacion': 4}
}

# Crear base de datos SQLite
def init_db():
    conn = sqlite3.connect('historial.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS solicitudes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TIMESTAMP,
        nombre TEXT,
        genero TEXT,
        edad TEXT,
        region TEXT,
        income REAL,
        monto REAL,
        plazo INTEGER,
        credit_worthiness TEXT,
        property_value REAL,
        dti REAL,
        ltv REAL,
        score_cliente REAL,
        riesgo_difuso REAL,
        probabilidad REAL,
        score_final REAL,
        decision TEXT,
        motivo TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

def calcular_dti(loan_amount, income, term):
    """
    Calcula Debt-to-Income Ratio.
    DTI = (Cuota Mensual / Ingreso Mensual) * 100
    """
    if income == 0:
        return 100
    
    # Cuota mensual aproximada (asumiendo tasa 8% anual)
    tasa_mensual = 0.08 / 12
    cuota_mensual = (loan_amount * tasa_mensual * (1 + tasa_mensual)**term) / ((1 + tasa_mensual)**term - 1)
    
    dti = (cuota_mensual / income) * 100
    return round(dti, 2)

def calcular_ltv(loan_amount, property_value):
    """
    Calcula Loan-to-Value Ratio.
    LTV = (Monto Préstamo / Valor Propiedad) * 100
    """
    if property_value == 0:
        return 100
    
    ltv = (loan_amount / property_value) * 100
    return round(ltv, 2)

def calcular_score_cliente(data):
    """
    Sistema de puntuación del perfil del cliente (0-100).
    Mayor score = Menor riesgo
    """
    score = 100  # Empezar con puntuación perfecta
    
    # ============ CREDIT WORTHINESS (35% del score) ============
    worthiness = data.get('credit_worthiness', 'Fair')
    if worthiness == 'Poor':
        score -= 50  # Penalización severa
    elif worthiness == 'Fair':
        score -= 30
    elif worthiness == 'Good':
        score -= 15
    elif worthiness == 'Excellent':
        score -= 0
    
    # ============ DTI RATIO (30% del score) ============
    dti = float(data.get('dti', 0))
    if dti > 50:  # Más del 50% del ingreso va al préstamo
        score -= 35
    elif dti > 40:
        score -= 25
    elif dti > 30:
        score -= 15
    elif dti > 20:
        score -= 8
    
    # ============ LTV RATIO (20% del score) ============
    ltv = float(data.get('ltv', 0))
    if ltv > 95:  # Casi sin enganche
        score -= 30
    elif ltv > 85:
        score -= 20
    elif ltv > 75:
        score -= 12
    elif ltv > 65:
        score -= 5
    
    # ============ ESTRUCTURA DEL PRÉSTAMO (10% del score) ============
    if data.get('neg_amortization', False):
        score -= 20  # Deuda que crece = muy peligroso
    
    if data.get('interest_only', False):
        score -= 15  # No paga capital
    
    if data.get('lump_sum_payment', False):
        score -= 12  # Pago único al final = riesgo
    
    if data.get('business_or_commercial', False):
        score -= 10  # Créditos comerciales más riesgosos
    
    # ============ HISTORIAL Y RESPALDO (5% del score) ============
    if not data.get('approv_in_adv', False):
        score -= 8  # Sin aprobación previa
    
    if not data.get('co_applicant', False):
        score -= 5  # Sin co-solicitante = menos respaldo
    
    # Garantizar que esté entre 0-100
    return max(0, min(100, score))

def generar_respuesta(score_cliente, riesgo_difuso, score_final, probabilidad, decision, motivo, data):
    """Genera respuesta, guarda en SQLite y en Firebase"""
    
    # ====== GUARDAR EN SQLITE ======
    try:
        conn = sqlite3.connect('historial.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO solicitudes 
            (fecha, nombre, genero, edad, region, income, monto, plazo, credit_worthiness, 
             property_value, dti, ltv, score_cliente, riesgo_difuso, probabilidad, 
             score_final, decision, motivo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
             data.get('nombre', 'Sin nombre'),
             data.get('gender', 'Male'),
             data.get('age', '35-44'),
             data.get('region', 'Central'),
             float(data.get('income', 0)),
             float(data.get('loan_amount', 0)),
             int(data.get('term', 360)),
             data.get('credit_worthiness', 'Fair'),
             float(data.get('property_value', 0)),
             float(data.get('dti', 0)),
             float(data.get('ltv', 0)),
             float(score_cliente),
             float(riesgo_difuso), 
             float(probabilidad),
             float(score_final),
             str(decision),
             str(motivo)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error al guardar en BD: {e}")
    
    # ====== GUARDAR TAMBIÉN EN FIREBASE ======
    # Aquí sí usamos la función que definiste arriba
    guardar_evaluacion_en_firebase(
        entrada=data,
        resultado_modelo={
            "score_cliente": float(score_cliente),
            "probabilidad": float(probabilidad),
            "score_final": float(score_final),
            "decision": str(decision),
            "motivo": str(motivo)
        },
        riesgo_difuso=float(riesgo_difuso)
    )
    
    # ====== RESPUESTA AL FRONT ======
    return jsonify({
        'score_cliente': round(float(score_cliente), 2),
        'riesgo_difuso': round(float(riesgo_difuso), 2),
        'probabilidad': round(float(probabilidad), 2),
        'score_final': round(float(score_final), 2),
        'dti': round(float(data.get('dti', 0)), 2),
        'ltv': round(float(data.get('ltv', 0)), 2),
        'decision': str(decision),
        'motivo': str(motivo)
    })


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validar que data no sea None
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        # Validar que todos los campos requeridos estén presentes
        required_fields = ['loan_amount', 'income', 'term', 'inflacion', 'combustible', 
                          'protestas', 'desempleo', 'covid', 'clima', 'credit_worthiness', 
                          'gender', 'age', 'region', 'credit_type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido faltante: {field}'}), 400
            if data[field] is None or data[field] == '':
                return jsonify({'error': f'Campo {field} está vacío'}), 400
        
        # ============ PASO 1: CALCULAR MÉTRICAS FINANCIERAS ============
        try:
            loan_amount = float(data['loan_amount'])
            income = float(data['income'])
            term = int(data['term'])
            property_value = float(data.get('property_value', loan_amount * 1.2))
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Valor numérico inválido: {str(e)}'}), 400
        
        dti = calcular_dti(loan_amount, income, term)
        ltv = calcular_ltv(loan_amount, property_value)
        
        data['dti'] = dti
        data['ltv'] = ltv
        
        # ============ PASO 2: CALCULAR SCORE DEL CLIENTE ============
        score_cliente = calcular_score_cliente(data)
        
        # ============ PASO 3: CALCULAR RIESGO DIFUSO (MACROECONOMÍA) ============
        try:
            riesgo_externo = calcular_riesgo_difuso(
                float(data['inflacion']), 
                float(data['combustible']),
                float(data['protestas']), 
                float(data['desempleo']),
                float(data['covid']), 
                float(data['clima'])
            )
        except Exception as e:
            print(f"Error en cálculo difuso: {e}")
            riesgo_externo = 5.0  # Valor por defecto en caso de error
        
        # ============ PASO 4: BANDERAS ROJAS (RECHAZO AUTOMÁTICO) ============
        
        # BANDERAS CRÍTICAS DEL CLIENTE
        if data.get('credit_worthiness') == 'Poor':
            return generar_respuesta(score_cliente, riesgo_externo, 25.0, 95.0, 
                                    "RECHAZADO", "Historial Crediticio Deficiente", data)
        
        if dti > 55:
            return generar_respuesta(score_cliente, riesgo_externo, 30.0, 90.0,
                                    "RECHAZADO", "DTI Crítico - Capacidad de Pago Insuficiente", data)
        
        if ltv > 98:
            return generar_respuesta(score_cliente, riesgo_externo, 28.0, 88.0,
                                    "RECHAZADO", "LTV Crítico - Colateral Insuficiente", data)
        
        if data.get('neg_amortization', False):
            return generar_respuesta(score_cliente, riesgo_externo, 20.0, 92.0,
                                    "RECHAZADO", "Amortización Negativa No Permitida", data)
        
        if income < (loan_amount / 180):  # No puede pagar ni en 15 años
            return generar_respuesta(score_cliente, riesgo_externo, 22.0, 94.0,
                                    "RECHAZADO", "Ingresos Insuficientes para el Monto Solicitado", data)
        
        # BANDERAS CRÍTICAS MACROECONÓMICAS
        if float(data['inflacion']) > 50:
            return generar_respuesta(score_cliente, riesgo_externo, 25.0, 85.0,
                                    "RECHAZADO", "Crisis Inflacionaria - Entorno Económico Crítico", data)
        
        if float(data['desempleo']) > 15:
            return generar_respuesta(score_cliente, riesgo_externo, 27.0, 83.0,
                                    "RECHAZADO", "Desempleo Crítico - Recesión Severa", data)
        
        if float(data['covid']) > 7000:
            return generar_respuesta(score_cliente, riesgo_externo, 26.0, 82.0,
                                    "RECHAZADO", "Colapso Sanitario - Alerta Crítica", data)
        
        if float(data['protestas']) > 3500:
            return generar_respuesta(score_cliente, riesgo_externo, 28.0, 80.0,
                                    "RECHAZADO", "Inestabilidad Social Severa", data)
        
        if float(data['combustible']) > 6.0:
            return generar_respuesta(score_cliente, riesgo_externo, 29.0, 81.0,
                                    "RECHAZADO", "Crisis Energética Crítica", data)
        
        # Combinación letal: estanflación
        if float(data['inflacion']) > 20 and float(data['desempleo']) > 10:
            return generar_respuesta(score_cliente, riesgo_externo, 22.0, 88.0,
                                    "RECHAZADO", "Estanflación Severa Detectada", data)
        
        # ============ PASO 5: RECHAZO POR SCORE BAJO ============
        if score_cliente < 35:
            return generar_respuesta(score_cliente, riesgo_externo, 30.0, 85.0,
                                    "RECHAZADO", "Perfil Crediticio Crítico", data)
        
        # ============ PASO 6: CONSTRUIR FEATURES PARA RANDOM FOREST ============
        try:
            gender_encoded = mappings['Gender'][data['gender']]
            age_encoded = mappings['age'][data['age']]
            region_encoded = mappings['Region'][data['region']]
            credit_type_encoded = mappings['credit_type'][data['credit_type']]
        except KeyError as e:
            return jsonify({'error': f'Valor inválido en campo categórico: {str(e)}'}), 400
        
        features = [
            gender_encoded,
            age_encoded,
            region_encoded,
            credit_type_encoded,
            float(data['income']),
            float(data['loan_amount']),
            int(data['term']),
            float(data['inflacion']),
            float(data['combustible']),
            float(data['protestas']),
            float(data['covid']),
            float(data['desempleo']),
            float(data['clima'])
        ]
        
        # One-hot encoding
        gender_onehot = [0, 0, 0]
        gender_onehot[gender_encoded] = 1
        features.extend(gender_onehot)
        
        age_onehot = [0] * 7
        age_onehot[age_encoded] = 1
        features.extend(age_onehot)
        
        region_onehot = [0, 0, 0, 0]
        region_onehot[region_encoded] = 1
        features.extend(region_onehot)
        
        credit_onehot = [0, 0, 0]
        credit_onehot[credit_type_encoded] = 1
        features.extend(credit_onehot)
        
        # Ajustar a 32 features
        features = features[:32]
        while len(features) < 32:
            features.append(0)
        
        # ============ PASO 7: PREDICCIÓN RANDOM FOREST ============
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prob_base = float(modelo.predict_proba(features_scaled)[0][1])
        
        # ============ PASO 8: AJUSTES Y PENALIZACIONES ============
        
        # Penalización por riesgo macroeconómico
        if riesgo_externo > 8.0:
            prob_base += 0.40
        elif riesgo_externo > 6.5:
            prob_base += 0.25
        elif riesgo_externo > 5.0:
            prob_base += 0.15
        elif riesgo_externo > 3.5:
            prob_base += 0.08
        
        # Penalización por DTI alto
        if dti > 45:
            prob_base += 0.20
        elif dti > 35:
            prob_base += 0.12
        
        # Penalización por LTV alto
        if ltv > 90:
            prob_base += 0.15
        elif ltv > 80:
            prob_base += 0.08
        
        # Penalización por estructura riesgosa
        if data.get('interest_only', False):
            prob_base += 0.12
        if data.get('lump_sum_payment', False):
            prob_base += 0.10
        if data.get('business_or_commercial', False):
            prob_base += 0.08
        
        prob = min(100.0, prob_base * 100)
        
        # ============ PASO 9: SCORE FINAL COMBINADO ============
        # 40% Perfil Cliente + 30% Entorno + 30% Modelo ML
        score_final = (
            (score_cliente * 0.40) +
            ((10 - riesgo_externo) * 10 * 0.30) +
            ((100 - prob) * 0.30)
        )
        
        # ============ PASO 10: DECISIÓN FINAL ============
        decision = "APROBADO"
        motivo = "Perfil de Riesgo Aceptable"
        
        if score_final < 45:
            decision = "RECHAZADO"
            motivo = f"Score Final Bajo ({score_final:.1f}/100)"
        elif score_final < 60:
            decision = "REVISIÓN MANUAL"
            motivo = f"Score Límite ({score_final:.1f}/100) - Requiere Análisis Adicional"
        elif prob > 45:
            decision = "RECHAZADO"
            motivo = f"Alta Probabilidad de Incumplimiento ({prob:.1f}%)"
        elif score_cliente < 50:
            decision = "REVISIÓN MANUAL"
            motivo = "Perfil de Cliente Requiere Evaluación Detallada"
        else:
            decision = "APROBADO"
            motivo = f"Perfil de Riesgo Aceptable - Score: {score_final:.1f}/100"
        
        return generar_respuesta(score_cliente, riesgo_externo, score_final, prob, 
                                decision, motivo, data)
        
    except KeyError as e:
        return jsonify({'error': f'Campo faltante: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Valor inválido: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/historial')
def historial():
    try:
        conn = sqlite3.connect('historial.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM solicitudes ORDER BY fecha DESC LIMIT 50')
        rows = cursor.fetchall()
        conn.close()
        
        historial_list = []
        for row in rows:
            historial_list.append({
                'id': row[0],
                'fecha': row[1],
                'nombre': row[2],
                'monto': row[7],
                'credit_worthiness': row[9],
                'dti': row[11],
                'ltv': row[12],
                'score_cliente': row[13],
                'riesgo_difuso': row[14],
                'probabilidad': row[15],
                'score_final': row[16],
                'decision': row[17]
            })
        
        return jsonify({'historial': historial_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/estadisticas')
def estadisticas():
    try:
        conn = sqlite3.connect('historial.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM solicitudes')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM solicitudes WHERE decision = "APROBADO"')
        aprobadas = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM solicitudes WHERE decision = "RECHAZADO"')
        rechazadas = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM solicitudes WHERE decision = "REVISIÓN MANUAL"')
        revision = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(score_cliente) FROM solicitudes')
        score_prom = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(riesgo_difuso) FROM solicitudes')
        riesgo_prom = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(probabilidad) FROM solicitudes')
        prob_prom = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(dti) FROM solicitudes')
        dti_prom = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(ltv) FROM solicitudes')
        ltv_prom = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total': total,
            'aprobadas': aprobadas,
            'rechazadas': rechazadas,
            'revision': revision,
            'score_cliente_promedio': round(score_prom, 2),
            'riesgo_promedio': round(riesgo_prom, 2),
            'probabilidad_promedio': round(prob_prom, 2),
            'dti_promedio': round(dti_prom, 2),
            'ltv_promedio': round(ltv_prom, 2),
            'tasa_aprobacion': round((aprobadas / total * 100) if total > 0 else 0, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/test')
def test():
    try:
        db.collection("pruebas").document("test1").set({"estado": "ok"})
        return "Conexión Firestore funcionando"
    except Exception as e:
        return f"Error -> {e}"


if __name__ == '__main__':
    app.run(debug=True, port=5000)
