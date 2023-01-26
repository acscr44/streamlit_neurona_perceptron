import streamlit as st
import numpy as np
import math


class Neuron:
    def __init__(self, weights, bias, func):
        self.pesos = weights
        self.sesgo = bias
        self.func_activ = func

    def salida(self, input_data):
        input_data = np.array(np.float_(input_data))
        return np.dot(input_data, self.pesos) + self.sesgo

    def run(self, input_data):
        input_data = self.salida(input_data)
        if self.func_activ == "sigmoid":
            return Neuron.sigmoid(input_data)
        elif self.func_activ == "relu":
            return Neuron.relu(input_data)
        elif self.func_activ == "tanh":
            return Neuron.tanhip(input_data)

    def change_bias(self, value):
        self.sesgo = value

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.e ** (-x))

    @staticmethod
    def relu(x):
        if x < 0:
            return 0
        else:
            return x

    @staticmethod
    def tanhip(x):  # No permite definirlo como método privado, tampoco admite el nombrado "tanh"
        return math.tanh(x)


st.set_page_config(layout="wide")

# Heading Image
st.image("./image.jpg")

# Title
st.title("¡Hola neurona!")
st.write("")

st.header("Simulador de neurona")

# Selector de pesos/entradas
qty_neurona = st.slider('Elige el número de entradas/pesos que tendrá la neurona', 1, 10, key='qty_neurona')

# Estructura de columnas para Pesos
st.subheader('Pesos')
col_pesos = st.columns(qty_neurona)
var_peso = []  # Variable Pesos
key_id_peso = []
for c in range(qty_neurona):
    with col_pesos[c]:
        i = c

        # for i in range(qty_neurona):
        var_peso.append('peso_w' + str(i))  # La variable_peso es definida con un str.
        key_id_peso.append(var_peso[i])  # Variable_peso es reasignado a con una key_id_peso
        var_peso[i] = st.number_input('Peso ' + str(i + 1), key=str(key_id_peso[i]))
st.write("w = ", str(var_peso))

# Estructura de columnas para Entradas
st.subheader('Entradas')
col_entradas = st.columns(qty_neurona)
var_entrada = []  # Variable Entradas
key_id_entrada = []
for c in range(qty_neurona):
    with col_entradas[c]:
        e = c

        var_entrada.append('entrada_x' + str(e))  # La variable_peso es definida con un str.
        key_id_entrada.append(var_entrada[e])  # Variable_peso es reasignado a con una key_id_peso
        var_entrada[e] = st.number_input('Entrada ' + str(e + 1), key=str(key_id_entrada[e]))
st.write("x = ", str(var_entrada))

col1, col2 = st.columns(2)

with col1:
    st.subheader('Sesgo')
    sesgo = st.number_input('Introduce el valor del sesgo', key="sesgo")
with col2:
    st.subheader('Función de activación')
    ml_model = ["Sigmoide", "ReLU", "Tangente hiperbólica"]  # Mantener idéntica correlación entre estas dos listas
    ml_func_activ = ["sigmoid", "relu", "tanh"]  # Mantener idéntica correlación entre estas dos listas
    opcion = st.selectbox('Elige la función de activación', ml_model, key="sesgo2")
    func_activ = ml_func_activ[ml_model.index(opcion)]


# If button is pressed
if st.button("Calcular la salida", key='submit'):
    # Calculo de resultado

    # salida = peso_t31 * entrada_t31 + peso_t32 * entrada_t32 + peso_t33 * entrada_t33 + bias_t3
    # salida = np.dot(var_peso, var_entrada)
    p1 = Neuron(var_peso, sesgo, func_activ)
    salida = p1.run(var_entrada)

    st.text(f"La salida de la neurona es {salida}")
