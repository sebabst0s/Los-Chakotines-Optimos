---
title: Los Chakotines Optimos
emoji: 🎯
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# Los Chakotines Óptimos

Herramienta web de optimización numérica con métodos de gradiente y condiciones de Wolfe.

## Métodos implementados

| Método | Descripción |
|--------|-------------|
| Gradiente Descendente | Steepest Descent con búsqueda de línea Wolfe |
| Gradiente Conjugado   | Fletcher-Reeves con reinicio cada n pasos |
| Método de Newton      | Hessiano numérico + Cholesky modificado |

## Características

- Funciones arbitrarias de n variables (sintaxis Python/NumPy)
- Comparación simultánea de los 3 métodos
- Gráficos de convergencia interactivos (Chart.js)
- Camino de convergencia animado con isolíneas (n=2)
- Parámetros de Wolfe configurables (c1, c2, α₀)

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Abrir `http://localhost:7860` en el navegador.

## Estructura

```
.
├── app.py            # Backend Flask + algoritmos
├── templates/
│   └── index.html    # Interfaz de usuario (SPA)
├── requirements.txt
├── Dockerfile
└── README.md
```
