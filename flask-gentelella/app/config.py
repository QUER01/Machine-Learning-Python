class Config(object):
    # the values of those depend on your setup
    APPLICATION_NAME ="My Application"
    POSTGRES_URL = "192.168.1.109:32776"
    POSTGRES_USER = "postgres"
    POSTGRES_PW = ""
    POSTGRES_DB = "webapp"
    POSTGRES_SCHEMA_USERS = "gentelella"
    POSTGRES_SCHEMA_DATA = "data"
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=POSTGRES_USER, pw=POSTGRES_PW, url=POSTGRES_URL,
                                                                   db=POSTGRES_DB)



    SECRET_KEY = 'key'
    #SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    DEBUG = False


class DebugConfig(Config):
    DEBUG = True
