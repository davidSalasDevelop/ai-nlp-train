# test.py
import mlflow
import pandas

# La prueba más simple del mundo
if __name__ == "__main__":
    with mlflow.start_run():
        print("--- ¡HOLA MUNDO DESDE EL SERVIDOR! ---")
        mlflow.log_metric("test_checkpoint", 1)
        mlflow.set_tag("status", "Test successful")
        print("--- ¡EL TEST HA FUNCIONADO! ---")
