from fastapi import FastAPI
import joblib
import uvicorn

app = FastAPI()


@app.get("/pred")
def predict(age: int, gender: int):
    
    model = joblib.load('music-recommender.joblib')

    prediction = model.predict([[age, gender]])
    
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=True)
