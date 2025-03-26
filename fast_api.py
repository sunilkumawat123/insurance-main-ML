from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import pickle
import os
import ssl
import uvicorn

app = FastAPI()

# Load Model Once at Startup
current_path = os.getcwd()
model_path = os.path.join(current_path, "model/best_model.pkl")
with open(model_path, "rb") as file:
    pipeline = pickle.load(file)

# Set up Template Directory
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/forms", response_class=HTMLResponse)
async def forms(request: Request):
    return templates.TemplateResponse("forms.html", {"request": request})


@app.post("/submit_data")
async def submit_data(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...)
):
    try:
        data = [[age, sex, bmi, children, smoker, region]]
        input_ = pd.DataFrame(data, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

        # Make Prediction
        pred = pipeline.predict(input_)

        return templates.TemplateResponse(
            "result.html",
            {"request": request, "prediction": f"${pred[0]:.2f}"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain("cert.pem", "key.pem")

    uvicorn.run(app, host="0.0.0.0", port=4000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
