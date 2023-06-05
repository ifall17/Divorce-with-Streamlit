import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


#Load Dataset
df = pd.read_csv("divorce_data.csv",sep = ';')
ts = pd.read_csv("reference.tsv", sep = '\t')



def main ():
        st.title("Divorce Prediction ")
        repons = []
        for index, row in ts.iterrows():
                question = row[0]
                res = st.radio(f"Question {index+1} : {question}", options=['Never','Rarely','Average','Frequently','Always'],
                index = 2, horizontal=True)
                if res == 'Never':
                        res = 0
                elif res == 'Rarely':
                        res = 1
                elif res == 'Average':
                        res = 2
                elif res == 'Frequently':
                        res = 3
                elif res == 'Always':
                        res = 4
                repons.append(res)

        if st.button("Send"):
                response = np.array(repons).reshape(1,-1)
                print(len(response), response.shape, response)
                # print(len(df.drop('Divorce', axis=1)),(df.drop('Divorce', axis=1).shape))
                # Transformer la liste en un tableau à deux dimensions
                # response_df = pd.DataFrame(response,columns=df.columns[:-1]) # Créer un DataFrame à partir du tableau
                #
                model = LogisticRegression()
                model.fit(df.drop('Divorce', axis=1).values, df['Divorce'])
                prediction = model.predict(response)

                st.subheader("Résultat de la prédiction :")
                st.write("Prédiction :", prediction[0])


#st.sidebar.success("Ibra")

if __name__=='__main__':
        main()


