##projeto para ler uma tabela e dividir os dados - automatizacao de tarefas
import pandas as pd
import re

#leitura da base 
contatos = pd.read_excel(r'C:\Users\mia.visantos\Downloads\proj\contacts_brochure.xls')

#definindo a equacao
def split_data(data):
    
    #definindo o padrao a ser descoberto
    pais_padrao = r'\b(?:Brazil|Chile|Argentina|Peru|Coloombia|St Kittis and Nevis|Jamaica|)\b'
    email_padrao = r'\b][A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.[A-Z|a-z]{2,}\b'
    telefone_padrao = r'\b\d{2,4}[-.\s]??\d{2,4}[-.\s]??\d{4,}\b'

#findall pra categorizar as colunas
    pais = re.findall(pais_padrao, data)
    email = re.findall(email_padrao, data)
    telefone = re.findall(telefone_padrao, data)
    
    nome = re.sub(pais_padrao,'', data)
    nome = re.sub(email_padrao, '', data)
    nome = re.sub(telefone_padrao, '', data)
    
    return pais [0] if pais else '', nome,email[0] if email else '', telefone[0] if telefone else ''

        
contatos[['Pais', 'Nome', 'Email', 'Telefone']] = contatos.apply(lambda row: split_data(row['Dados']), axis=1, result_type='expand')

contatos.to_excel('contatos_brochures.xls', index=False)

print(f'Dados salvos com sucesso')
    
