# Fine-tuning de Qwen3 para Affekt_core - Iteration Compact

Estado: PROPUESTA. Los umbrales de "Success metric" y "Guardrails" son mios y hay que confirmarlos antes de entrenar.

**Goal:** que Qwen3-0.6B (y 1.7B como comparacion) diga la linea del NPC con el tono que pide el motor, sin ejemplos en el prompt.

**Who cares:** el jugador (NPC que suena vivo y coherente); el desarrollador (latencia baja en CPU local).

**Decision owner:** Aaron.

**Action changed by the model:** `AffektEngine._generate_response` pasa de "prompt con 3 ejemplos + modelo base" a "prompt sin ejemplos + modelo ajustado". El motor determinista (vector, memoria, grafo, triggers) no cambia: sigue decidiendo ANIMO y HECHOS; el modelo solo redacta.

**Success metric (sobre escenarios reservados, ver abajo):**
- tono >= 85 % (heuristica `test/tone.py`) y >= 80 % aceptable en revision manual de 30 muestras
- latencia mediana <= 2.0 s en tu CPU, sin few-shot

**Guardrail metrics (no pueden empeorar frente al modelo base):**
- hechos (precio/objeto correcto) >= 98 %
- respuestas cortas (<= 3 frases, <= 220 caracteres) >= 98 %
- sin lenguaje meta ("the player", "mood", "style") >= 98 %
- variedad entre repeticiones >= 85 %
- sin copiar respuestas del dataset de entrenamiento (4-gramas) >= 95 %
- cero acotaciones (`*...*`, corchetes, parentesis)

**Mistake budget / unacceptable mistakes:** precio u objeto equivocado; romper el personaje (hablar del jugador en tercera persona o mencionar el sistema); contenido fuera de ficcion de juego.
**Acceptable mistakes:** frases planas pero correctas; tono ligeramente suave.

**Assumptions:**
- el motor determinista elige bien el animo (el modelo no lo corrige)
- la heuristica de tono es solo orientativa (ya vimos falsos PASS/FAIL), por eso la revision manual
- 2.000-3.000 ejemplos limpios bastan para LoRA en modelos de este tamano (a validar)

**Constraints:** CPU local para inferencia (llama.cpp, GGUF); entrenamiento en GPU gratuita de la nube (T4); licencia del modelo profesor revisada antes de entrenar con sus salidas.

**Labels and data snapshot:**
- Entrada: payload exacto que produce `AffektEngine.build_payload` (misma estructura que en juego).
- Salida: linea del NPC generada por el profesor y filtrada.
- Profesor: nvidia/nemotron-3-ultra-550b-a55b:free via OpenRouter (`--teacher openrouter`), elegido porque se espera una
  licencia que permita entrenar con sus salidas. PENDIENTE DE VERIFICAR por Aaron en la ficha del modelo; la licencia
  se registra en cada fila (`--teacher-license`). TBrilliance 2.6B queda como alternativa local (14 s/respuesta en CPU).
- Escenarios: rejilla de NPC (rol/nombre) x animo x accion x objeto/precio. Los objetos y precios varian ampliamente; incluir tambien los del juego.
- Filtro: reutilizar `run_checks` + rechazar acotaciones, tercera persona, > 3 frases, casi-duplicados.
- Version del dataset: hash del archivo + semilla de generacion guardados junto al modelo.

**Split policy (anti-fuga):**
- Los 15 escenarios de `test/eval_llm.py` NO entran al entrenamiento.
- Reservar tambien NPC/roles y objetos que el entrenamiento nunca vea.
- Deduplicar entre entrenamiento y evaluacion.

**Train/serve parity:** el prompt de entrenamiento = `build_messages` de inferencia con 0 ejemplos (incluido `/no_think` para Qwen3). Si cambia el prompt de inferencia, hay que reentrenar. Tarea pendiente: parametro `shots` en `build_messages`.

**Baseline (modelo base, 5 repeticiones, semilla 1, con 3 ejemplos en el prompt):**

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Global | 94 % | 96 % |
| Tono | 62 % | 84 % |
| Hechos | 100 % | 100 % |
| Variedad | 99 % | 88 % |
| Latencia | 3.4 s | 5.4 s |

**Eval slices:** por animo (hostil / amistoso / neutro / tenso), por accion (comprar / atacar / saludar / ayudar / preguntar), por NPC.

**Known risks:**
- el modelo aprende los defectos del profesor (acotaciones, rarezas); por eso el filtro y la revision manual
- sobreajuste a frases del dataset (copia); por eso la metrica de 4-gramas
- el tono medido por palabras clave puede premiar vocabulario, no actitud
- el fine-tuning puede olvidar capacidad general; irrelevante para este uso pero conviene comprobar que no genera basura fuera de distribucion

**Next experiment:** generar 200 ejemplos de prueba con TBrilliance, filtrar y leer 30 a mano para validar el filtro antes de escalar.

**Rollback or fallback:** conservar el GGUF base. `load_backend` ya cae al stub si el modelo no carga; la ruta del modelo es configurable, asi que volver atras es cambiar la ruta. El nuevo modelo solo se adopta si cumple TODOS los umbrales sobre los escenarios reservados.
