# GigaChain агент выполняющий код
Проект на GigaChain где реализован REPL агент, которому подается на вход
задача и он с помощью написания кода, пытается её решить.
Если в процессе решения задачи GigaChat написал код, который
выдает ошибку, то мы отправляем ошибку обратно к llm и просим
переписать код. Этот процесс происходит итеративно.
## Настройка 
### Выполнение кода
1. Можно запустить в докер контейнере с командой `docker compose up` в папке
jupyter_fastapi
2. Можно запустить локально с помощью следующих команд `poetry install` и `make run`
### Запуск streamlit демо с агентом
#### ENV настройка
В папке streamlit_demo/app скопировать файл `.env-example` в `.env`
Назначить нужные переменные

В этом файле есть следующие переменные:

**GIGACHAT_USER** — юзернейм

**GIGACHAT_PASSWORD** — пароль

**GIGACHAT_BASE_URL** — урл для API гигачата, если использовать через credentials можно не заполнять

**GIGACHAT_CREDENTIALS** — креды 

**GIGACHAT_SCOPE** — scope для кредов (GIGACHAT_API_PERS или GIGACHAT_API_CORP)

**JUPYTER_CLIENT_API** — ссылка на API где будет выполняться код

#### Запуск
1. `poetry install`
2. В папке `streamlit_demo` запустить `poetry run streamlit run main.py`
