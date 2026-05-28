# OptimNum — Optimización Numérica

Aplicación web para encontrar mínimos de funciones mediante tres métodos clásicos,
todos con búsqueda de línea que satisface las **condiciones de Wolfe** (Armijo + curvatura).

## Métodos implementados

| Método | Descripción |
|--------|-------------|
| Gradiente descendente | Steepest descent con búsqueda de línea Wolfe |
| Gradiente conjugado   | Fletcher-Reeves, con reinicio cada *n* pasos |
| Método de Newton      | Hessiano numérico + Cholesky modificado para regularización |

La búsqueda de línea implementa el **Algoritmo 3.5/3.6 de Nocedal & Wright**
(zoom con condiciones de Wolfe fuertes).

---

## Ejecución local

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Levantar servidor de desarrollo
python app.py
```

Abrir `http://localhost:5000` en el navegador.

---

## Deploy en Render

### Opción A — desde el dashboard de Render

1. Ir a [render.com](https://render.com) → **New → Web Service**.
2. Conectar el repositorio de GitHub/GitLab que contiene este proyecto.
3. Render detecta automáticamente el `Procfile`; revisar que los campos sean:

   | Campo | Valor |
   |-------|-------|
   | **Environment** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app` |

4. Hacer clic en **Create Web Service**.
5. En 2-3 minutos el servicio estará disponible en `https://<nombre>.onrender.com`.

### Opción B — Render Blueprint (render.yaml)

Crear `render.yaml` en la raíz del proyecto:

```yaml
services:
  - type: web
    name: optimnum
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

Subir al repositorio y en el dashboard de Render elegir **New → Blueprint**.

---

## Estructura del proyecto

```
.
├── app.py               # Backend Flask + algoritmos de optimización
├── templates/
│   └── index.html       # SPA — interfaz de usuario
├── requirements.txt     # Dependencias Python
├── Procfile             # Comando de arranque para Render/Heroku
└── README.md
```

---

## Uso de la interfaz

1. **Ejemplos rápidos** — carga funciones de prueba predefinidas (Rosenbrock, Himmelblau…).
2. **Función objetivo** — cualquier expresión Python; usa `x[0]`, `x[1]`… y funciones de `numpy` (`np.sin`, `np.exp`, etc.).
3. **Punto inicial** — valores separados por comas, uno por variable.
4. **Parámetros de Wolfe** — `0 < c1 < c2 < 1`; valores típicos: `c1=1e-4`, `c2=0.9`.
5. Pulsar **Optimizar** (o `Ctrl+Enter`).

Los resultados muestran: punto mínimo, valor de la función, iteraciones, norma del gradiente, gráfico de convergencia e historial completo.
