from app.api import app
from mangum import Mangum

handler = Mangum(app)  # required for Vercel’s serverless functions
