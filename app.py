# Import libraries
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

# Create app object
app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Loading ml model
trained_model = pickle.load(open('salary_data_model_dtr_r2_972.pkl', 'rb'))

# Get user input
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Salary prediction API
@app.post('/predict', response_model=dict)
async def predict_salary(request: Request):
    data = await request.form()
    age = int(data['age'])
    gender = data['gender']
    if(gender=="female"):
        female = 1
        male = 0
    else:
        female = 0
        male = 1
    education_level = data['education_level']
    job_title = data['job_title']
    yoe = data['years_of_experience'] 
    salary = trained_model.predict([[age, female, male, education_level, job_title, yoe]])
    salary = "$"+str(int(salary[0]))
    print()
    return {'salary': salary}

# Run the API
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)