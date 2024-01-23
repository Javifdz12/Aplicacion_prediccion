from flask import Flask, render_template, jsonify, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model("Pesos1_PracticaAA.best.hdf5")

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para predecir
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos de la solicitud
        data = request.get_json(force=True)
        user_sequence = np.array([data['user_sequence']])  # Asegúrate de enviar la secuencia

        # Loguear la entrada
        print("Entrada al modelo:")
        print(user_sequence)

        # Realizar la predicción
        prediction = model.predict(user_sequence)  # Hacer la predicción

        # Loguear la salida
        print("Salida del modelo:")
        print(prediction)

        # Enviar la predicción como respuesta
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
