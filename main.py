import uvicorn

if __name__ == "__main__":
    # Запускаем приложение из модуля app.api
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000)