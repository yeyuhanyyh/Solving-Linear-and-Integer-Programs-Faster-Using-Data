"""Run the current IP experiments. Historical commands have the legacy_ prefix."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parent/'current'
sys.path.insert(0,str(root))
if __name__=='__main__':runpy.run_path(str(root/'run.py'),run_name='__main__')
