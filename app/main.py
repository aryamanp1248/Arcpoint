from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Arcpoint Context API")
app.include_router(router)
