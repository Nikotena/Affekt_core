# Kankei Core

Motor narrativo para videojuegos construido bajo una **filosofía 80/20**: 80% lógica determinista, 20% inteligencia artificial. El núcleo determinista simula la emoción, memoria y contexto de escena de los NPC; un micro-LLM local (AMD Ryzen AI / ROCm) se reserva como capa de refinamiento que convierte el estado estructurado en lenguaje natural.

## Módulos

| Módulo | Archivo | Responsabilidad |
|--------|---------|-----------------|
| Personalidad Vectorial (4D) | `modules/VectorialPersonality.py` | Física emocional `V = (X, Y, Z, W)` |
| Memoria (Memory Set) | `modules/MemorySet.py` | Memoria de 3 capas (corto / medio / largo plazo) |
| Función de Contexto | `modules/ContextFunction.py` | Orquestador de escena + payload estricto para LLM |
| Gestión Neo4j | `modules/GraphDatabase.py` | Grafo en memoria + persistencia Neo4j opcional |
| Controlador del Motor | `KankeiEngine.py` | Coordina la tubería completa |

## Requisitos

- **Python 3.8+** (desarrollado y probado en Python 3.12)
- No se requieren dependencias de terceros. El driver `neo4j` es **opcional** — si no está instalado, el grafo funciona solo en memoria.

Verifica tu instalación de Python:

```powershell
python --version
```

---

## Cómo Probar — Paso a Paso

### Paso 1: Abre una terminal en la carpeta del proyecto

```powershell
cd D:\Kankei\Kankei_core
```

> En PowerShell 5.1, **no uses `&&`** para encadenar comandos. Usa `;` en su lugar.

### Paso 2: Ejecuta la suite completa de pruebas

Esto ejecuta 6 pruebas independientes (cableado del motor, matemáticas de personalidad, memoria, contexto, grafo, tubería completa) y luego una simulación de 20 acciones con métricas.

```powershell
python test\basic_tests.py
```

### Paso 2b: Ejecuta las pruebas unitarias (comportamiento)

```powershell
python -m unittest discover -s test -v
```

### Paso 3: Ejecuta la demo del motor directamente

```powershell
python KankeiEngine.py
```

Salida esperada (resumida):

```
Kankei Engine initialized successfully
...
=== Engine Results ===
Action 1: NPC_0 - 'greet the NPC'
  Payload size: 2 tokens
  Graph subgraph: 4 nodes, 5 edges
  Memory event persisted: event_NPC_0_1
...
In-memory graph: 27 nodes
Neo4j persistence active: False

Engine shutdown complete.
```

### Paso 4: Ejecuta con un número personalizado de acciones

```powershell
python test\basic_tests.py --actions 50
```

---

## Qué Verifica Cada Prueba

| Prueba | Verifica |
|--------|-----------|
| Test 1 | El motor arranca y todos los módulos están conectados |
| Test 2 | El vector de personalidad 4D se actualiza y se mantiene en rangos válidos |
| Test 3 | El buffer circular de memoria corta almacena/recupera por NPC aislado |
| Test 4 | La detección de triggers dispara por palabras clave y construye un payload para LLM |
| Test 5 | Nodos/aristas del grafo, búsqueda de vecinos, recorrido de subgrafo, exportación Cypher |
| Test 6 | La tubería completa de acción produce respuestas + eventos del grafo |

La suite debe terminar con:

```
=== All Basic Tests Passed ===
```

y un bloque de métricas:

```
=== Engine Metrics ===
Total actions: 20
Average payload size: ~10 lines
Max payload size: ~14 lines
Memory events persisted: 16
Graph nodes: 27
Neo4j persistence active: False
```

---

## Arranque rápido con Gemma 4 (demo / servidor)

```python
from KankeiEngine import create_engine

engine = create_engine()                 # busca el modelo en ./model, CPU
engine = create_engine(n_gpu_layers=-1)  # en la nube con GPU (llama-cpp-python compilado con CUDA)
r = engine.process_action("NPC_1", "Hey Marta, how much for a healing potion?")
print(r["response"], r["intent"], r["mood"])
```

- El modelo se busca en este orden: argumento `model_path=`, variable de entorno `KANKEI_MODEL`,
  y `./model/**/gemma-4-E2B-it-Q4_0.gguf` (o cualquier `gemma-4*.gguf` de más de 500 MB, para
  ignorar descargas cortadas). Si no hay modelo, lanza un error claro en vez de responder con
  frases enlatadas.
- Capas en GPU: argumento `n_gpu_layers=` o variable `KANKEI_GPU_LAYERS`.
- Configuración medida como la mejor: 2 ejemplos por llamada tomados de 5 conjuntos fijos por
  ánimo (el comienzo del prompt se repite y `llama.cpp` lo reutiliza), temperatura 0.5,
  `top_p` 0.9, `top_k` 40, y el precio pedido explícitamente en compras y regateos.
- `create_engine(llm=StubBackend())` crea el motor sin modelo (pruebas).
- El primer turno de cada combinación (ánimo, conjunto de ejemplos) es más lento mientras se
  llena la caché. `create_engine(warm_up=True)` la llena al arrancar (19 prefijos, unos 90 s en
  CPU, segundos con GPU; la caché ocupa ~80 MB) y baja el turno medio de 5.7 s a 3.7 s en CPU.
  Recomendado en el servidor.

## Texto libre del jugador

El motor entiende lo que el jugador escribe con `modules/IntentParser.py`, un analizador por
reglas (sin modelo, determinista). Devuelve la intención (saludo, pregunta, compra, regateo,
elogio, insulto, queja, amenaza, ataque...), cuánto cambia el ánimo del NPC, el precio
ofrecido y el objeto mencionado.

- El jugador puede dirigirse al NPC por su nombre o por un apodo (`Hey Marta, ...`,
  `[NPC_Name]`); el nombre se ignora para entender la frase. Apodos nuevos:
  `engine.add_alias("NPC_1", "Big M")`.
- Medir cuánto acierta con frases reales: escribe una frase por línea en un `.txt` y

  ```powershell
  python test\review_lines.py text.txt --out lines_review.csv
  ```

  Abre el CSV y rellena `fix_intent` (y `fix_valence`: -1, 0 o 1) **solo en las filas que
  el analizador falló**. Después:

  ```powershell
  python test\review_lines.py --score lines_review.csv --save-labels labels.json
  ```

  Muestra el acierto, qué reglas causan los errores y las confusiones más frecuentes.
- Bancos de prueba: `python test\eval_intent.py --set dev|heldout|stress|user|all --errors`.

## Activar la Persistencia Neo4j (Opcional)

Por defecto el grafo es **solo en memoria**, que es todo lo necesario
para ejecutar las pruebas. Para persistir el grafo a una instancia Neo4j:

1. Instala el driver:

   ```powershell
   pip install neo4j
   ```

2. Inicia una instancia de Neo4j (o usa Neo4j Desktop / Docker):

   ```powershell
   docker run -d --name neo4j -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5
   ```

3. Instancia el motor con la conexión habilitada:

   ```python
   from KankeiEngine import KankeiEngine

   engine = KankeiEngine(
       neo4j_uri="bolt://localhost:7687",
       neo4j_user="neo4j",
       neo4j_password="password",
       connect_neo4j=True,
   )
   engine.initialize_modules()
   engine.running = True
   engine.run(20)
   engine.shutdown()   # sincroniza el grafo de vuelta a Neo4j
   ```

`engine.graph.is_persistent()` devuelve `True` cuando hay una conexión activa.

## Exportación Offline del Grafo

Puedes volcar todo el grafo como un lote de instrucciones Cypher para análisis offline:

```python
engine.save_snapshot("world.cypher")
```

---

## Solución de Problemas

| Síntoma | Causa / Solución |
|---------|------------------|
| `python : The token '&&' is not a valid statement separator` | Estás en PowerShell 5.1. Usa `;` para encadenar comandos. |
| `ModuleNotFoundError: No module named 'neo4j'` | Esperado a menos que hayas optado por persistencia Neo4j. El motor funciona en memoria; instala el driver solo para persistencia. |
| `ModuleNotFoundError: No module named 'modules'` | Ejecuta los comandos desde `D:\Kankei\Kankei_core` (el archivo de pruebas ya agrega la ruta). |
| Errores de Unicode/emoji en la consola | La suite de pruebas usa marcadores ASCII `[PASS]` / `[FAIL]` para evitar problemas de códec en Windows. |