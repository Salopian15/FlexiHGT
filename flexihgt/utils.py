import os
import subprocess
import sys
import importlib.util
# Check to see if dependencies are installed

# Check to see if diamond and/or mmseqs are installed
def check_diamond_mmseqs():
    """
    Check for the presence of diamond and mmseqs in the PATH.
    """
    # Check for diamond
    try:
        subprocess.run(['diamond', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        print('diamond not found in PATH. Please install diamond and add it to your PATH.')
        return False
    try:
        subprocess.run(['mmseqs', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        print('mmseqs not found in PATH. Please install mmseqs and add it to your PATH.')
        return False

def check_dependencies():
    """
    Check for python dependencies.
    """
    # check for biopython, pandas, ete3 and numpy presence and versions
    if importlib.util.find_spec('biopython') is None:
        print('biopython not found. Please install biopython.')
        return False
    if importlib.util.find_spec('pandas') is None:
        print('pandas not found. Please install pandas.')
        return False
    if importlib.util.find_spec('ete3') is None:
        print('ete3 not found. Please install ete3.')
        return False
    if importlib.util.find_spec('numpy') is None:
        print('numpy not found. Please install numpy.')
        return False

def check_ete3db():
    if os.path.exists(os.path.expanduser('~\\.etetoolkit\\taxa.sqlite')):
        return True
    elif os.path.exists(os.path.expanduser('~/.etetoolkit/taxa.sqlite')):
        return True
    else:
        print('ete3 database not found. Please run ete3 upgrade to download the database.')
        return False

def check_all():
    """
    Check for all dependencies.
    """
    check_diamond_mmseqs()
    check_dependencies()
    # check if ete3 returns True
    if check_ete3db() and check_diamond_mmseqs() and check_dependencies():
        print('All dependencies are installed.')
    else:
        print('Please install the missing dependencies and re-run the program.')
        # show which dependencies are missing
        if not check_diamond_mmseqs():
            print('diamond and/or mmseqs are missing.')
        if not check_dependencies():
            print('biopython, pandas, ete3 and numpy are missing.')
        if not check_ete3db():
            print('ete3 database is missing.')
        sys.exit(1)