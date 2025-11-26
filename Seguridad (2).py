import numpy as np
import pandas as pd

RUTA_CSV = "DIRECCION DEL ARCHIVO CSV AQUI.csv"
S_BASE_MVA = 100.0 
pd.set_option('display.precision', 5)
pd.set_option('display.width', 200)

def cargar_datos():
    """Carga los datos del sistema desde el archivo CSV."""
    try:
        raw_df = pd.read_csv(RUTA_CSV)
        raw_df.columns = raw_df.columns.str.strip().str.replace(r'\s+|\r|\n', '', regex=True)
    except FileNotFoundError:
        print(f"Error FATAL: No se encontró el archivo CSV en la ruta: {RUTA_CSV}")
        return None, None, None
    
    line_cols = ['FromBus', 'Tobus', 'X(pu)', 'P0(MW)From']
    line_df = raw_df[line_cols].copy().dropna(subset=['FromBus', 'Tobus', 'X(pu)'])
    line_df['FromBus'] = line_df['FromBus'].astype(int)
    line_df['Tobus'] = line_df['Tobus'].astype(int)
    line_df['LineName'] = line_df['FromBus'].astype(str) + '-' + line_df['Tobus'].astype(str)
    
    bus_cols = ['Bus', 'Pgen', 'Pload']
    bus_df = raw_df[bus_cols].copy().dropna(subset=['Bus'])
    bus_df['Bus'] = bus_df['Bus'].astype(int)
    bus_df[['Pgen', 'Pload']] = bus_df[['Pgen', 'Pload']].fillna(0)
    
    P_net_base_pu = (bus_df['Pgen'] - bus_df['Pload']).values
    
    print("--- Datos del Sistema Cargados Exitosamente ---")
    print(f"{len(line_df)} líneas y {len(bus_df)} buses encontrados.")
    return line_df, bus_df, P_net_base_pu

def construir_matrices_base(line_df, bus_df):
    """Construye las matrices B' y F del caso base."""
    buses_totales = len(bus_df)
    B_base = np.zeros((buses_totales, buses_totales))
    line_df['b_series'] = 1.0 / line_df['X(pu)']
    
    for _, branch in line_df.iterrows():
        k_idx, m_idx = int(branch['FromBus']) - 1, int(branch['Tobus']) - 1
        b = branch['b_series']
        B_base[k_idx, k_idx] += b
        B_base[m_idx, m_idx] += b
        B_base[k_idx, m_idx] -= b
        B_base[m_idx, k_idx] -= b
        
    B_reducida = B_base[1:, 1:]
    try:
        F_matrix_reducida = np.linalg.inv(B_reducida)
    except np.linalg.LinAlgError:
        print("Error FATAL: La matriz B base es singular. El sistema no está conectado.")
        return None, None
        
    F_matrix_completa = np.zeros((buses_totales, buses_totales))
    F_matrix_completa[1:, 1:] = F_matrix_reducida
    
    print("--- Matrices B y F Calculadas ---")
    print("B :")
    print(B_base)
    print("\nF :")
    print(F_matrix_completa)
    return B_base, F_matrix_completa

def calcular_sensibilidades(line_df, bus_df, F_matrix_completa):
    """
    Pre-calcula las matrices de sensibilidad GSF y LODF.
    """
    NL = len(line_df)
    TM = len(bus_df)
    line_names = line_df['LineName'].values
    bus_numbers = bus_df['Bus'].astype(str).values
    
    gsf_matrix = np.zeros((NL, TM))
    for k_idx, line in line_df.iterrows():
        v_idx = int(line['FromBus']) - 1
        w_idx = int(line['Tobus']) - 1
        X_vw = line['X(pu)']
        
        for i_idx in range(TM):
            F_vi = F_matrix_completa[v_idx, i_idx]
            F_wi = F_matrix_completa[w_idx, i_idx]
            gsf_matrix[k_idx, i_idx] = (F_vi - F_wi) / X_vw
            
    gsf_df = pd.DataFrame(gsf_matrix, index=line_names, columns=[f"Bus_{b}" for b in bus_numbers])

    lodf_matrix = np.zeros((NL, NL))
    for l_idx, branch_l in line_df.iterrows(): 
        k_falla, m_falla = int(branch_l['FromBus']) - 1, int(branch_l['Tobus']) - 1
        x_l = branch_l['X(pu)']
        
        denominador = x_l - (F_matrix_completa[k_falla, k_falla] + F_matrix_completa[m_falla, m_falla] - 2 * F_matrix_completa[k_falla, m_falla])
        if np.isclose(denominador, 0): continue

        for k_idx, branch_k in line_df.iterrows(): 
            if l_idx == k_idx:
                lodf_matrix[k_idx, l_idx] = -1.0 
                continue
            
            v_mon, w_mon = int(branch_k['FromBus']) - 1, int(branch_k['Tobus']) - 1
            x_k = branch_k['X(pu)']
            
            numerador_parcial = (F_matrix_completa[v_mon, k_falla] - F_matrix_completa[v_mon, m_falla]) - \
                                (F_matrix_completa[w_mon, k_falla] - F_matrix_completa[w_mon, m_falla])
            
            d_kl = (numerador_parcial / denominador) * (x_l / x_k)
            lodf_matrix[k_idx, l_idx] = d_kl
            
    lodf_df = pd.DataFrame(lodf_matrix, index=line_names, columns=line_names)
    
    print("--- Matrices de Sensibilidad (GSF y LODF) Calculadas ---")
    return gsf_df, lodf_df

def calcular_flujos_dc_mw(theta_completo, line_df):
    """Calcula los flujos de potencia (MW) para un vector de ángulos dado."""
    flujos = {}
    for _, branch in line_df.iterrows():
        k_idx = int(branch['FromBus']) - 1
        m_idx = int(branch['Tobus']) - 1
        flujo_pu = (theta_completo[k_idx] - theta_completo[m_idx]) / branch['b_series']
        flujos[branch['LineName']] = flujo_pu * S_BASE_MVA
    return flujos

def analizar_contingencia_nk(fallas_gen, fallas_linea, line_df_base, B_base, P_net_base, bus_df):
    """
    Analiza una contingencia N-k (múltiples fallas) RECALCULANDO la matriz B.
    """
    print("  Contingencia N-k detectada. Usando recálculo completo de matriz B...")
    buses_totales = len(bus_df)
    
    P_cont = P_net_base.copy()
    slack_bus_idx = 0 
    
    for gen_num in fallas_gen:
        gen_idx = gen_num - 1
        p_out = bus_df.loc[gen_idx, 'Pgen']
        if p_out > 0:
            P_cont[gen_idx] -= p_out 
            P_cont[slack_bus_idx] += p_out 
            print(f"  Falla Gen {gen_num}: {p_out*S_BASE_MVA:.1f} MW perdidos, compensados por Slack.")
        
    B_cont = B_base.copy()
    lineas_en_servicio = line_df_base.copy()
    
    for line_name in fallas_linea:
        branch = line_df_base[line_df_base['LineName'] == line_name].iloc[0]
        k_idx, m_idx = int(branch['FromBus']) - 1, int(branch['Tobus']) - 1
        b = branch['b_series']
        
        B_cont[k_idx, k_idx] -= b
        B_cont[m_idx, m_idx] -= b
        B_cont[k_idx, m_idx] += b
        B_cont[m_idx, k_idx] += b
        print(f"  Falla Línea {line_name}: Línea removida de la topología.")
        lineas_en_servicio = lineas_en_servicio[lineas_en_servicio['LineName'] != line_name]

    B_cont_reducida = B_cont[1:, 1:]
    P_cont_reducido = P_cont[1:]
    
    try:
        F_cont_reducida = np.linalg.inv(B_cont_reducida)
        theta_cont_reducido = F_cont_reducida @ P_cont_reducido
        theta_cont_completo = np.insert(theta_cont_reducido, 0, 0)
        
        flujos_post_falla = calcular_flujos_dc_mw(theta_cont_completo, lineas_en_servicio)
        
        for line_name in fallas_linea:
            flujos_post_falla[line_name] = 0.0
        
        return flujos_post_falla

    except np.linalg.LinAlgError:
        print("\nError Fatal: La contingencia ha creado una 'isla' (matriz singular).")
        return None

if __name__ == "__main__":

    line_df, bus_df, P_net_base_pu = cargar_datos()
    if line_df is None:
        exit()
        
    B_base, F_matrix_completa = construir_matrices_base(line_df, bus_df)
    if B_base is None:
        exit()
    gsf_df, lodf_df = calcular_sensibilidades(line_df, bus_df, F_matrix_completa)
    print("\n" + "="*70)
    print("MATRIZ DE SENSIBILIDAD GSF")
    print("Filas: Líneas monitoreadas / Columnas: Bus de inyección")
    print("="*70)
    print(gsf_df)
    
    print("\n" + "="*70)
    print("MATRIZ DE SENSIBILIDAD LODF")
    print("Filas: Líneas monitoreadas / Columnas: Línea en falla")
    print("="*70)
    print(lodf_df)

    flujos_base_ac = line_df.set_index('LineName')['P0(MW)From'].to_dict()
    df_resultados_base = pd.DataFrame.from_dict(flujos_base_ac, orient='index', columns=['P_Base (MW)'])

    while True:
        print("\n" + "="*70)
        print("CONTINGENCIAS")
        print("="*70)
        print("Generadores con Pgen > 0:", list(bus_df[bus_df['Pgen'] > 0]['Bus']))
        print("Líneas (ejemplos): 1-2, 1-4, 1-5, 2-3, 2-4, ...")
        
        entrada_usuario = input(
            "\nIntroduce las fallas separadas por comas (o Enter para salir):\n"
            "(Ej. N-1: g2) o (Ej. N-1: l1-4)\n"
            "(Ej. N-k: g2, l1-4, l2-3)\n"
            "Tu entrada: "
        )
        
        if not entrada_usuario.strip():
            print("--- Saliendo del simulador ---")
            break
            
        fallas_gen_num = []
        fallas_linea_str = []
        elementos_falla = [e.strip().lower() for e in entrada_usuario.split(',')]
        
        for item in elementos_falla:
            if item.startswith('g'):
                try: fallas_gen_num.append(int(item[1:]))
                except: print(f"Ignorando entrada de Gen inválida: '{item}'")
            elif item.startswith('l'):
                fallas_linea_str.append(item[1:])
            elif '-' in item: 
                fallas_linea_str.append(item)
            else:
                print(f"Advertencia: Entrada '{item}' no reconocida y será ignorada.")
        
        fallas_linea_str = [f for f in fallas_linea_str if f in lodf_df.index]
        fallas_gen_num = [g for g in fallas_gen_num if g in bus_df['Bus'].values and bus_df.loc[g-1, 'Pgen'] > 0]
        
        total_fallas = len(fallas_gen_num) + len(fallas_linea_str)
        df_resultados_final = df_resultados_base.copy()
        
        if total_fallas == 0:
            print("No se introdujeron fallas válidas. Mostrando caso base.")
            print(df_resultados_final.to_string())
            continue
            
        
        if total_fallas == 1:
            print("--- Analizando (N-1) usando Factores de Sensibilidad (GSF/LODF) ---")
            
            if fallas_gen_num:
                gen_num = fallas_gen_num[0]
                gen_idx = gen_num - 1
                p_falla_pu = -bus_df.loc[gen_idx, 'Pgen'] 
                print(f"Falla Gen {gen_num}: {p_falla_pu * S_BASE_MVA:.1f} MW")
                
                gsf_vector = gsf_df[f"Bus_{gen_num}"]
                cambio_mw = gsf_vector * p_falla_pu * S_BASE_MVA
                
                df_resultados_final['Cambio (MW)'] = cambio_mw
                df_resultados_final['P_Nuevo (MW)'] = df_resultados_final['P_Base (MW)'] + cambio_mw
            
            else:
                linea_falla = fallas_linea_str[0]
                p_falla_mw = flujos_base_ac[linea_falla] 
                print(f"Falla Línea {linea_falla}: {p_falla_mw:.1f} MW")
                
                lodf_vector = lodf_df[linea_falla]
                cambio_mw = lodf_vector * p_falla_mw
                
                df_resultados_final['Cambio (MW)'] = cambio_mw
                df_resultados_final['P_Nuevo (MW)'] = df_resultados_final['P_Base (MW)'] + cambio_mw
                df_resultados_final.loc[linea_falla, 'P_Nuevo (MW)'] = 0.0

        else:
            print("--- Analizando (N-k) usando Recálculo de Matriz B ---")
            
            flujos_post_falla = analizar_contingencia_nk(
                fallas_gen_num, fallas_linea_str, line_df, B_base, P_net_base_pu, bus_df
            )
            
            if flujos_post_falla:
                df_resultados_final['P_Nuevo (MW)'] = df_resultados_final.index.map(flujos_post_falla).fillna(0.0)
                df_resultados_final['Cambio (MW)'] = df_resultados_final['P_Nuevo (MW)'] - df_resultados_final['P_Base (MW)']
            else:
                print("El análisis N-k falló (posible isla).")
                continue
        print("\n--- Resultados de Flujo de Potencia (MW) ---")
        print(df_resultados_final.to_string(float_format="%.3f"))