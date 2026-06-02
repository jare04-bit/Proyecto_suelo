from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Cargar el modelo y el escalador con sus nombres correctos
modelo = joblib.load('modelo_xgb_cafe.pkl')
scaler = joblib.load('escalador_cafe.pkl')

# Código HTML incrustado con diseño mejorado, colores café y soporte de imágenes dinámicas
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
            background: linear-gradient(135deg, #e6d5c3 0%, #c3a68f 100%);
            background-attachment: fixed;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(74, 59, 50, 0.2);
            max-width: 650px;
            width: 100%;
            overflow: hidden;
        }
        .banner-header {
            width: 100%;
            height: 180px;
            object-fit: cover;
            display: block;
        }
        .content {
            padding: 30px;
        }
        h1 { 
            color: #4a3b32; 
            text-align: center; 
            margin-top: 10px;
            margin-bottom: 25px;
            font-size: 28px;
            font-weight: 700;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .form-group { display: flex; flex-direction: column; }
        label { margin-bottom: 6px; font-weight: 600; color: #5a4a42; font-size: 14px; }
        input {
            padding: 11px;
            border: 2px solid #dcd1c4;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
            background-color: #fdfbf7;
        }
        input:focus {
            outline: none;
            border-color: #8c6239;
        }
        button {
            grid-column: span 2;
            background-color: #5c3a21;
            color: white;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
            transition: background 0.3s, transform 0.1s;
            box-shadow: 0 4px 10px rgba(92, 58, 33, 0.3);
        }
        button:hover { background-color: #442a17; }
        button:active { transform: scale(0.99); }
        
        .result {
            margin-top: 30px;
            border-radius: 12px;
            overflow: hidden;
            display: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            background-color: white;
            border: 1px solid #e0e0e0;
        }
        .result-img {
            width: 100%;
            height: 140px;
            object-fit: cover;
        }
        .result-body {
            padding: 20px;
        }
        .apto { border-left: 6px solid #28a745; }
        .apto h3 { color: #1e7e34; margin-top: 0; }
        .no-apto { border-left: 6px solid #dc3545; }
        .no-apto h3 { color: #bd2130; margin-top: 0; }
        
        .certeza-text {
            font-weight: 600;
            color: #666;
            margin-bottom: 15px;
        }
        ul { padding-left: 20px; color: #444; }
        li { margin-bottom: 8px; line-height: 1.4; }
    </style>
</head>
<body>

<div class="container">
    <img src="https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?q=80&w=800&auto=format&fit=crop" class="banner-header" alt="Cafetal">
    
    <div class="content">
        <h1>☕ Evaluador Inteligente de Suelos</h1>
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
            <button type="submit">📋 Analizar Muestra de Tierra</button>
        </form>

        <div id="resultadoBox" class="result">
            <img id="resultadoImg" src="" class="result-img" alt="Estado del terreno">
            <div class="result-body">
                <h3 id="resultadoTitulo"></h3>
                <p id="resultadoCerteza" class="certeza-text"></p>
                <h4>💡 Recomendaciones Técnicas:</h4>
                <div id="recomendacionesTexto"></div>
            </div>
        </div>
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
    const resImg = document.getElementById('resultadoImg');
    
    resBox.style.display = 'block';
    
    // Configuración visual dinámica según la predicción del modelo
    if (res.clase === 1) {
        resBox.className = 'result apto';
        // Imagen de planta de café saludable creciendo
        resImg.src = "https://images.unsplash.com/photo-1509042239860-f550ce710b93?q=80&w=600&auto=format&fit=crop";
    } else {
        resBox.className = 'result no-apto';
        // Imagen de tierra seca/inadecuada
        resImg.src = "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?q=80&w=600&auto=format&fit=crop";
    }
    
    document.getElementById('resultadoTitulo').innerText = res.diagnostico;
    document.getElementById('resultadoCerteza').innerText = `Fiabilidad del diagnóstico: ${res.certeza}%`;
    
    let recsHtml = '<ul>';
    res.recomendaciones.forEach(r => recsHtml += `<li>${r}</li>`);
    recsHtml += '</ul>';
    document.getElementById('recomendacionesTexto').innerHTML = recsHtml;
    
    // Desplazar la pantalla suavemente hacia el resultado
    resBox.scrollIntoView({ behavior: 'smooth' });
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
        diagnostico = "🎉 ¡SUELO OPTIMIZADO Y APTO PARA CAFÉ!"
        recs = [
            "**Preservar el balance:** Las condiciones actuales son excelentes. Evita añadir fertilizantes nitrogenados innecesarios para no quemar la raíz.",
            "**Conservación de humedad:** El nivel hídrico reportado es ideal. Se aconseja implementar coberturas orgánicas (acolchado) para protegerlo de la evaporación solar directa.",
            "**Control preventivo:** El pH se ubica en el punto dulce para la absorción de nutrientes del café. Realiza calicatas de control cada ciclo productivo."
        ]
    else:
        diagnostico = "❌ CONDICIONES NO APTAS PARA EL CULTIVO"
        recs = [
            "**Plan de Enmiendas Químicas:** Tus macronutrientes esenciales (N-P-K) no se alinean con las demandas del cultivo de café. Diseña un plan de fertilización correctiva basada en estos desbalances.",
            "**Monitoreo del Microclima:** Los factores de temperatura o lluvia están limitando el éxito del cultivo. Considera la implementación de sistemas de riego controlado o el uso de árboles de sombra (como el género Inga) para regular la temperatura.",
            "**Alternativa Agronómica:** Las características de este suelo sugieren una afinidad natural hacia otro tipo de bioma. Si las correcciones son muy costosas, evalúa la rotación de cultivos compatibles."
        ]
        
    return jsonify({
        'clase': prediccion,
        'diagnostico': diagnostico,
        'certeza': f"{certeza:.2f}",
        'recomendaciones': recs
    })

if __name__ == '__main__':