from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

modelo = joblib.load('modelo_xgb_cafe.pkl')
scaler = joblib.load('escalador_cafe.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluador de Suelo - Café</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #4a3b32;
            margin: 0;
            padding: 50px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .titulo-superior {
            color: #ffffff;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 30px;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
            letter-spacing: 0.5px;
        }
        
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            max-width: 750px;
            width: 100%;
        }
        h1 { 
            color: #4a3b32; 
            text-align: left; 
            margin-bottom: 30px; 
            font-size: 20px;
            border-bottom: 2px solid #6f4e37;
            padding-bottom: 10px;
            letter-spacing: 1px;
        }
        
        .seccion-formulario {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .form-group { 
            display: flex; 
            flex-direction: column; 
            background-color: #fdfbf9;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #eae5e1;
            transition: all 0.3s ease;
        }
        .form-group:focus-within {
            border-color: #6f4e37;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(111, 78, 55, 0.1);
        }
        
        label { 
            margin-bottom: 8px; 
            font-weight: bold; 
            color: #6f4e37; 
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input {
            padding: 8px 4px;
            border: none;
            border-bottom: 2px solid #ccc;
            border-radius: 0;
            font-size: 16px;
            background-color: transparent;
            color: #333;
            font-weight: 600;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-bottom-color: #6f4e37;
        }
        
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield;
        }

        /* Estructura del botón extendida */
        .btn-contenedor {
            display: flex;
            justify-content: flex-end;
            margin-top: 10px;
        }
        button {
            background-color: #6f4e37;
            color: white;
            padding: 14px 40px;
            border: none;
            border-radius: 30px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 10px rgba(111, 78, 55, 0.2);
        }
        button:hover { 
            background-color: #553c2a;
            transform: translateY(-1px);
        }
        button:active {
            transform: translateY(1px);
        }
        
        .result {
            margin-top: 35px;
            padding: 25px;
            border-radius: 16px;
            display: none;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
        }
        .apto { background-color: #bc804c; color: #000000; border: 1px solid #bc804c; }
        .no-apto { background-color: #f8d7da; color: #000000; border: 1px solid #f5c6cb; }
        .result h3 { margin-top: 0; font-size: 20px; }
        ul { padding-left: 20px; margin-top: 10px; }
        li { margin-bottom: 8px; line-height: 1.5; }
    </style>
</head>
<body>

<div class="titulo-superior">EVALUADOR DE TERRENO PARA CULTIVO DE CAFE</div>

<div class="container">
    <h1>DATOS DEL SUELO</h1>
    <form id="sueloForm">
        
        <div class="seccion-formulario">
            <div class="form-group">
                <label>Nitrógeno (N)</label>
                <input type="number" id="N" required step="any">
            </div>
            <div class="form-group">
                <label>Fósforo (P)</label>
                <input type="number" id="P" required step="any">
            </div>
            <div class="form-group">
                <label>Potasio (K)</label>
                <input type="number" id="K" required step="any">
            </div>
        </div>

        <div class="seccion-formulario">
            <div class="form-group">
                <label>pH del Suelo</label>
                <input type="number" id="ph" required step="any">
            </div>
            <div class="form-group">
                <label>Temperatura (°C)</label>
                <input type="number" id="temperature" required step="any">
            </div>
        </div>

        <div class="seccion-formulario">
            <div class="form-group">
                <label>Humedad (%)</label>
                <input type="number" id="humidity" required step="any">
            </div>
            <div class="form-group">
                <label>Precipitación / Lluvia (mm)</label>
                <input type="number" id="rainfall" required step="any">
            </div>
        </div>

        <div class="btn-contenedor">
            <button type="submit">Evaluar Terreno</button>
        </div>
    </form>

    <div id="resultadoBox" class="result">
        <h3 id="resultadoTitulo"></h3>
        <p id="resultadoCerteza"></p>
        <h4>Recomendaciones sobre el terreno:</h4>
        <div id="recomendacionesTexto"></div>
    </div>
</div>

<script>
document.getElementById('sueloForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const datos = {
        N: parseFloat(document.getElementById('N').value),
        P: parseFloat(document.getElementById('P').value),
        K: parseFloat(document.getElementById('K').value),
        ph: parseFloat(document.getElementById('ph').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        rainfall: parseFloat(document.getElementById('rainfall').value)
    };

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(datos)
    });
    
    const res = await response.json();
    
    const resBox = document.getElementById('resultadoBox');
    resBox.style.display = 'block';
    resBox.className = 'result ' + (res.clase === 1 ? 'apto' : 'no-apto');
    
    document.getElementById('resultadoTitulo').innerText = res.diagnostico;
    document.getElementById('resultadoCerteza').innerText = `Certeza del modelo: ${res.certeza}%`;
    
    let recsHtml = '<ul>';
    res.recomendaciones.forEach(r => recsHtml += `<li>${r}</li>`);
    recsHtml += '</ul>';
    document.getElementById('recomendacionesTexto').innerHTML = recsHtml;
});
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    df_usuario = pd.DataFrame([{
        'N': data['N'], 'P': data['P'], 'K': data['K'],
        'temperature': data['temperature'], 'humidity': data['humidity'], 
        'ph': data['ph'], 'rainfall': data['rainfall']
    }])
    
    datos_escalados = scaler.transform(df_usuario)
    prediccion = int(modelo.predict(datos_escalados)[0])
    probabilidades = modelo.predict_proba(datos_escalados)[0]
    certeza = float(probabilidades[prediccion] * 100)
    
    if prediccion == 1:
        diagnostico = "SUELO APTO PARA SU CULTIVO"
        recs = [
            "Mantener condiciones del suelo: las condiciones son óptimas evitar sobrecargar alguna cosa.",
            "Monitoreo de humedad: mantener humedad del terreno.",
            "Control de acidez: pH óptimo, mantener revisión del mismo"
        ]
    else:
        diagnostico = "SUELO NO APTO PARA CULTIVO"
        recs = [
            "Nutrientes: Problablemnete los niveles de N,P,K sean bajas.",
            "Clima: El clima puede no ser favirable par la cosecha.",
        ]
        
    return jsonify({
        'clase': prediccion,
        'diagnostico': diagnostico,
        'certeza': f"{certeza:.2f}",
        'recomendaciones': recs
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
