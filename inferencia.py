import pandas as pd
import pickle

def carregar_sistema():
    with open('pipeline_banco.pkl', 'rb') as f:
        return pickle.load(f)

def avaliar_cliente(dados_cliente_raw, sistema):
    modelo = sistema['modelo']
    scaler = sistema['scaler']
    dados_treino = sistema['dados_treino']
    dados_numericos = sistema['dados_numericos']
    dados_categoricos = sistema['dados_categoricos']
    
    df_cliente = pd.DataFrame([dados_cliente_raw])
    
    df_num = pd.DataFrame(scaler.transform(df_cliente[dados_numericos]), columns=dados_numericos)
    df_cat = pd.get_dummies(df_cliente[dados_categoricos].astype(str), prefix_sep='_', dtype=int)
    
    df_processado = df_num.join(df_cat)
    
    df_processado = df_processado.reindex(columns=dados_treino, fill_value=0)
    
    classe_predita = modelo.predict(df_processado)[0]
    probabilidades = modelo.predict_proba(df_processado)[0]
    
    prob_adimplente = probabilidades[0]
    prob_inadimplente = probabilidades[1]
    
    status = "Inadimplente (Default)" if abs(classe_predita - 1) < 0.01 else "Adimplente"
    
    print("\n=============================================")
    print("       RESULTADO DA ANÁLISE DE RISCO         ")
    print("=============================================")
    print(f"Classificação Final: Cliente será {status}")
    print(f"Score de Risco (Prob. de Default): {prob_inadimplente * 100:.2f}%")
    print("\n--- Distribuição de Probabilidade ---")
    print(f"Probabilidade de Pagar (Classe 0): {prob_adimplente * 100:.2f}%")
    print(f"Probabilidade de Não Pagar (Classe 1): {prob_inadimplente * 100:.2f}%")
    print("=============================================\n")

def main():
    novo_cliente = {
        'LIMIT_BAL': 50000,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 35,
        'PAY_0': 2, 'PAY_2': 2, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -2, 'PAY_6': -2,
        'BILL_AMT1': 3913, 'BILL_AMT2': 3102, 'BILL_AMT3': 689, 'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
        'PAY_AMT1': 0, 'PAY_AMT2': 689, 'PAY_AMT3': 0, 'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0
    }

    sistema = carregar_sistema()    
    avaliar_cliente(novo_cliente, sistema)

if __name__ == "__main__":    
    main()