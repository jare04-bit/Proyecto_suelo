from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Cargar el modelo y el escalador con sus nombres correctos
modelo = joblib.load('modelo_xgb_cafe.pkl')
scaler = joblib.load('escalador_cafe.pkl')

# Código HTML incrustado para mantenerlo todo en un solo script fácil de correr
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
            background-color: #f4f7f6;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 600px;
            width: 100%;
        }
        h1 { color: #4a3b32; text-align: center; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { margin-bottom: 5px; font-weight: bold; color: #555; }
        input {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 14px;
        }
        button {
            grid-column: span 2;
            background-color: #6f4e37;
            color: white;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
            transition: background 0.3s;
        }
        button:hover { background-color: #553c2a; }
        .result {
            margin-top: 25px;
            padding: 15px;
            border-radius: 6px;
            display: none;
        }
        .apto { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .no-apto { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        ul { padding-left: 20px; }
    </style>
</head>
<body>

<div class="container">
    <h1>☕ Evaluador de Suelos para Café</h1>
    <form id="sueloForm" class="grid">
        <div class="form-group">
            <label>Nitrógeno (N):</label>
            <input type="number" id="N" value="107" required step="any">
        </div>
        <div class="form-group">
            <label>Fósforo (P):</label>
            <input type="number" id="P" value="34" required step="any">
        </div>
        <div class="form-group">
            <label>Potasio (K):</label>
            <input type="number" id="K" value="32" required step="any">
        </div>
        <div class="form-group">
            <label>pH del Suelo:</label>
            <input type="number" id="ph" value="6.78" required step="any">
        </div>
        <div class="form-group">
            <label>Temperatura (°C):</label>
            <input type="number" id="temperature" value="26.77" required step="any">
        </div>
        <div class="form-group">
            <label>Humedad (%):</label>
            <input type="number" id="humidity" value="66.41" required step="any">
        </div>
        <div class="form-group" style="grid-column: span 2;">
            <label>Precipitación / Lluvia (mm):</label>
            <input type="number" id="rainfall" value="177.77" required step="any">
        </div>
        <button type="submit">📋 Evaluar Terreno</button>
    </form>

    <div id="resultadoBox" class="result">
        <h3 id="resultadoTitulo"></h3>
        <p id="resultadoCerteza"></p>
        <h4>💡 Recomendaciones:</h4>
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
        diagnostico = "🎉 ¡SUELO APTO PARA CAFÉ!"
        recs = [
            "Mantener el equilibrio: Las condiciones actuales son excelentes. Evita la sobrefertilización.",
            "Monitoreo hídrico: Conserva la humedad del suelo usando coberturas vegetales orgánicas.",
            "Control de acidez: Tu pH es óptimo, haz análisis preventivos cada ciclo."
        ]
    else:
        diagnostico = "❌ SUELO NO APTO PARA CAFÉ"
        recs = [
            "Revisión de Nutrientes: Compara tus niveles de N-P-K con las necesidades del café.",
            "Factor Climático: Si la temperatura o lluvia están fuera de rango, evalúa invernaderos o riego artificial.",
            "Alternativa: El ecosistema actual podría ser ideal para otro tipo de cultivo (rotación)."
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