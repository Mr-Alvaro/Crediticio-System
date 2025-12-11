import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def calcular_riesgo_difuso(inflacion, combustible, protestas, desempleo, covid, clima):
    """
    Calcula el riesgo externo usando lógica difusa con valores REALISTAS.
    
    Returns:
        float: Riesgo externo en escala 0-10
    """
    
    # Definir variables de entrada con rangos REALISTAS
    inflacion_var = ctrl.Antecedent(np.arange(0, 101, 1), 'inflacion')  # 0-100% (máximo realista)
    combustible_var = ctrl.Antecedent(np.arange(0, 8.1, 0.1), 'combustible')  # $0-8 (crisis extrema)
    protestas_var = ctrl.Antecedent(np.arange(0, 5001, 1), 'protestas')  # 0-5000 (más realista)
    desempleo_var = ctrl.Antecedent(np.arange(0, 25.1, 0.1), 'desempleo')  # 0-25% (depresión)
    covid_var = ctrl.Antecedent(np.arange(0, 10001, 1), 'covid')  # 0-10000 (pico pandemia)
    clima_var = ctrl.Antecedent(np.arange(-10, 41, 1), 'clima')  # -10 a 40°C
    
    # Variable de salida
    riesgo = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'riesgo')
    
    # INFLACIÓN (0-100%) - AJUSTADA A VALORES REALISTAS
    # Baja: 0-4% (óptimo)
    # Media: 3-10% (tolerable)
    # Alta: 8-25% (preocupante)
    # Muy Alta: 20-100% (crisis)
    inflacion_var['baja'] = fuzz.trapmf(inflacion_var.universe, [0, 0, 2, 5])
    inflacion_var['media'] = fuzz.trimf(inflacion_var.universe, [3, 6, 12])
    inflacion_var['alta'] = fuzz.trimf(inflacion_var.universe, [10, 18, 30])
    inflacion_var['muy_alta'] = fuzz.trapmf(inflacion_var.universe, [25, 50, 100, 100])
    
    # COMBUSTIBLE ($0-8) - MAYOR PESO EN DECISIONES
    # Bajo: $0-2.5 (económico)
    # Medio: $2-4 (normal)
    # Alto: $3.5-5.5 (caro)
    # Muy Alto: $5-8 (crisis energética)
    combustible_var['bajo'] = fuzz.trapmf(combustible_var.universe, [0, 0, 1.5, 2.5])
    combustible_var['medio'] = fuzz.trimf(combustible_var.universe, [2, 3, 4])
    combustible_var['alto'] = fuzz.trimf(combustible_var.universe, [3.5, 4.5, 5.5])
    combustible_var['muy_alto'] = fuzz.trapmf(combustible_var.universe, [5, 6.5, 8, 8])
    
    # PROTESTAS (0-5000) - AJUSTADO
    # Bajas: 0-500 (tranquilo)
    # Medias: 400-1500 (moderado)
    # Altas: 1200-3000 (preocupante)
    # Muy Altas: 2500-5000 (crisis social)
    protestas_var['bajas'] = fuzz.trapmf(protestas_var.universe, [0, 0, 300, 600])
    protestas_var['medias'] = fuzz.trimf(protestas_var.universe, [400, 900, 1600])
    protestas_var['altas'] = fuzz.trimf(protestas_var.universe, [1200, 2000, 3200])
    protestas_var['muy_altas'] = fuzz.trapmf(protestas_var.universe, [2500, 3500, 5000, 5000])
    
    # DESEMPLEO (0-25%) - CRÍTICO
    # Bajo: 0-5% (pleno empleo)
    # Medio: 4-8% (normal)
    # Alto: 7-15% (recesión)
    # Muy Alto: 12-25% (depresión)
    desempleo_var['bajo'] = fuzz.trapmf(desempleo_var.universe, [0, 0, 3, 5.5])
    desempleo_var['medio'] = fuzz.trimf(desempleo_var.universe, [4, 6, 9])
    desempleo_var['alto'] = fuzz.trimf(desempleo_var.universe, [7, 11, 16])
    desempleo_var['muy_alto'] = fuzz.trapmf(desempleo_var.universe, [12, 18, 25, 25])
    
    # COVID (0-10000) - AJUSTADO
    # Bajo: 0-1000 (controlado)
    # Medio: 800-3000 (moderado)
    # Alto: 2500-6000 (ola fuerte)
    # Muy Alto: 5000-10000 (colapso sanitario)
    covid_var['bajo'] = fuzz.trapmf(covid_var.universe, [0, 0, 600, 1200])
    covid_var['medio'] = fuzz.trimf(covid_var.universe, [800, 1800, 3500])
    covid_var['alto'] = fuzz.trimf(covid_var.universe, [2500, 4000, 6500])
    covid_var['muy_alto'] = fuzz.trapmf(covid_var.universe, [5000, 7500, 10000, 10000])
    
    # CLIMA - Sin cambios (menos crítico)
    clima_var['extremo_frio'] = fuzz.trapmf(clima_var.universe, [-10, -10, 2, 8])
    clima_var['normal'] = fuzz.trapmf(clima_var.universe, [5, 15, 25, 32])
    clima_var['extremo_calor'] = fuzz.trapmf(clima_var.universe, [28, 35, 40, 40])
    
    # RIESGO (salida) - REDEFINIDO
    riesgo['muy_bajo'] = fuzz.trapmf(riesgo.universe, [0, 0, 1, 2.5])
    riesgo['bajo'] = fuzz.trimf(riesgo.universe, [1.5, 3, 4.5])
    riesgo['medio'] = fuzz.trimf(riesgo.universe, [3.5, 5, 6.5])
    riesgo['alto'] = fuzz.trimf(riesgo.universe, [5.5, 7, 8.5])
    riesgo['muy_alto'] = fuzz.trapmf(riesgo.universe, [7.5, 9, 10, 10])
    
    # REGLAS DIFUSAS JERARQUIZADAS (Más realistas y sensibles)
    rules = [
        # ============ REGLAS CRÍTICAS (Riesgo MUY ALTO) ============
        # Crisis económica completa
        ctrl.Rule(inflacion_var['muy_alta'] | desempleo_var['muy_alto'], riesgo['muy_alto']),
        ctrl.Rule(protestas_var['muy_altas'], riesgo['muy_alto']),
        ctrl.Rule(covid_var['muy_alto'], riesgo['muy_alto']),
        
        # Estanflación (inflación + desempleo altos)
        ctrl.Rule(inflacion_var['alta'] & desempleo_var['alto'], riesgo['muy_alto']),
        
        # Crisis energética + económica
        ctrl.Rule(combustible_var['muy_alto'] & inflacion_var['alta'], riesgo['muy_alto']),
        ctrl.Rule(combustible_var['muy_alto'] & desempleo_var['alto'], riesgo['muy_alto']),
        
        # Combinaciones múltiples
        ctrl.Rule(inflacion_var['alta'] & protestas_var['altas'], riesgo['muy_alto']),
        ctrl.Rule(covid_var['alto'] & protestas_var['muy_altas'], riesgo['muy_alto']),
        
        # ============ REGLAS ALTO RIESGO ============
        ctrl.Rule(inflacion_var['alta'] & combustible_var['alto'], riesgo['alto']),
        ctrl.Rule(desempleo_var['alto'] & protestas_var['altas'], riesgo['alto']),
        ctrl.Rule(covid_var['alto'] & desempleo_var['medio'], riesgo['alto']),
        ctrl.Rule(combustible_var['alto'] & protestas_var['altas'], riesgo['alto']),
        ctrl.Rule(inflacion_var['media'] & desempleo_var['alto'], riesgo['alto']),
        
        # Combustible tiene PESO IMPORTANTE
        ctrl.Rule(combustible_var['muy_alto'], riesgo['alto']),
        
        # ============ REGLAS RIESGO MEDIO ============
        ctrl.Rule(inflacion_var['media'] & desempleo_var['medio'], riesgo['medio']),
        ctrl.Rule(combustible_var['medio'] & protestas_var['medias'], riesgo['medio']),
        ctrl.Rule(covid_var['medio'] & desempleo_var['medio'], riesgo['medio']),
        ctrl.Rule(inflacion_var['baja'] & protestas_var['altas'], riesgo['medio']),
        ctrl.Rule(clima_var['extremo_frio'] | clima_var['extremo_calor'], riesgo['medio']),
        ctrl.Rule(combustible_var['alto'] & inflacion_var['baja'], riesgo['medio']),
        
        # ============ REGLAS BAJO RIESGO ============
        ctrl.Rule(inflacion_var['baja'] & desempleo_var['bajo'] & protestas_var['bajas'], riesgo['bajo']),
        ctrl.Rule(inflacion_var['baja'] & covid_var['bajo'] & combustible_var['bajo'], riesgo['bajo']),
        ctrl.Rule(desempleo_var['bajo'] & protestas_var['bajas'] & clima_var['normal'], riesgo['muy_bajo']),
        ctrl.Rule(combustible_var['bajo'] & inflacion_var['baja'] & desempleo_var['bajo'], riesgo['muy_bajo']),
    ]
    
    # Sistema de control
    sistema_ctrl = ctrl.ControlSystem(rules)
    sistema = ctrl.ControlSystemSimulation(sistema_ctrl)
    
    # Asignar valores de entrada
    sistema.input['inflacion'] = min(inflacion, 100)  # Limitar a 100%
    sistema.input['combustible'] = min(combustible, 8)  # Limitar a $8
    sistema.input['protestas'] = min(protestas, 5000)  # Limitar a 5000
    sistema.input['desempleo'] = min(desempleo, 25)  # Limitar a 25%
    sistema.input['covid'] = min(covid, 10000)  # Limitar a 10000
    sistema.input['clima'] = max(-10, min(clima, 40))  # Limitar -10 a 40
    
    # Calcular
    sistema.compute()
    
    riesgo_calculado = sistema.output['riesgo']
    
    # ============ AJUSTES CRÍTICOS POST-CÁLCULO ============
    
    # Inflación muy alta (>50%) = Crisis automática
    if inflacion > 50:
        riesgo_calculado = max(riesgo_calculado, 9.0)
    elif inflacion > 30:
        riesgo_calculado = max(riesgo_calculado, 7.5)
    elif inflacion > 15:
        riesgo_calculado = max(riesgo_calculado, 6.0)
    
    # COVID crítico
    if covid > 7000:
        riesgo_calculado = max(riesgo_calculado, 8.5)
    elif covid > 5000:
        riesgo_calculado = max(riesgo_calculado, 7.0)
    
    # Protestas masivas
    if protestas > 3500:
        riesgo_calculado = max(riesgo_calculado, 8.0)
    elif protestas > 2000:
        riesgo_calculado = max(riesgo_calculado, 6.5)
    
    # Desempleo crítico
    if desempleo > 15:
        riesgo_calculado = max(riesgo_calculado, 8.5)
    elif desempleo > 10:
        riesgo_calculado = max(riesgo_calculado, 7.0)
    
    # Combustible crítico (MÁS SENSIBLE)
    if combustible > 6:
        riesgo_calculado = max(riesgo_calculado, 7.5)
    elif combustible > 5:
        riesgo_calculado = max(riesgo_calculado, 6.5)
    elif combustible > 4:
        riesgo_calculado = max(riesgo_calculado, 5.0)
    
    # Combinaciones mortales
    if inflacion > 20 and desempleo > 10:  # Estanflación severa
        riesgo_calculado = 9.5
    
    if combustible > 5 and inflacion > 15:  # Crisis energética + inflación
        riesgo_calculado = max(riesgo_calculado, 8.0)
    
    return round(min(riesgo_calculado, 10.0), 2)