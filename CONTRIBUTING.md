# Flujo de trabajo del equipo

1. Cada integrante crea su rama desde `main` y no modifica nombres de columnas ni etiquetas por su cuenta.
2. Antes de capturar, sincroniza `main`, activa `.venv` e instala `requirements.txt`.
3. Las decisiones compartidas se hacen en `config/catalogo.json`: tipo (`estatica` o `dinamica`) y `manos_requeridas` (`1` o `2`). Deben validarse con la referencia oficial antes de cambiarlas.
4. Cada referencia se guarda como `docs/referencias/<slug>.jpg`.
5. Los CSV locales se llaman `<slug>_<integrante>.csv`. No se mezclan archivos de 63 y 126 características.
6. Cada persona realiza varias sesiones con cambios controlados de distancia, iluminación y ángulo. No se capturan landmarks rojos o inestables.
7. Antes de integrar datos se ejecutan las pruebas y el consolidador.
8. Los datos pesados no se fusionan mediante Git; se entregan al responsable del dataset por el medio acordado y se validan antes de consolidar.

## Comandos

```powershell
.venv\Scripts\Activate.ps1
python -m src.data_collection.app_captura
python -m src.data_collection.verificar_entorno
python -m src.data_collection.consolidar_dataset
python -m unittest discover -s tests
```

Para una seña dinámica:

```powershell
python -m src.data_collection.recolector_secuencias geanella "por favor"
```
