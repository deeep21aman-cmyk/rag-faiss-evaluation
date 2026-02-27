from config import FILE_NAME

def load_data():
    try:
        with open(FILE_NAME,'r') as f:
            text=f.read()
            return text

    except (IsADirectoryError,FileNotFoundError) as e:
        raise Exception(f"File does not exist : {e}")
