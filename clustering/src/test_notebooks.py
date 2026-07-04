import json
import traceback
from pathlib import Path

def run_notebook(path):
    print(f"Probando {path}...")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code = ""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # evitar magic commands como %matplotlib inline
            lines = [line if not line.strip().startswith("%") else f"# {line}" for line in source.split("\n")]
            code += "\n".join(lines) + "\n\n"
            
    # Cambiamos el CWD temporalmente para simular estar en la raiz del proyecto
    import os
    original_cwd = os.getcwd()
    os.chdir(Path(__file__).parent.parent.parent)
    
    try:
        # We must use globals that has __builtins__
        exec_globals = {"__builtins__": __builtins__, "__name__": "__main__"}
        # Some cells might do plt.show() or display(), we can mock them
        code = "import matplotlib.pyplot as plt\nplt.show = lambda: None\ndef display(*args, **kwargs): print(*args)\n" + code
        exec(code, exec_globals)
        print(f"{path} OK! Ningún error.\n")
        res = True
    except Exception as e:
        print(f"Error en {path}:")
        print(traceback.format_exc())
        res = False
    finally:
        os.chdir(original_cwd)
    return res

if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).parent.parent
    t1 = run_notebook(base_dir / "notebooks" / "eda_tablas_v2.ipynb")
    t2 = run_notebook(base_dir / "notebooks" / "eda_graficos_v2.ipynb")
    if not (t1 and t2):
        sys.exit(1)
