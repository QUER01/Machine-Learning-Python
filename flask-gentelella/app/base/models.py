from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from config import Config
import datetime

class User(Base, UserMixin):

    __tablename__ = 'User'
    __table_args__ = {"schema": Config.POSTGRES_SCHEMA_USERS,
                      "extend_existing": True}

    id = Column(Integer, primary_key=True)
    username = Column(String(120), unique=True)
    email = Column(String(120), unique=True)
    password = Column(String(30))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)


class data_table_1(Base, UserMixin):

    __tablename__ = 'data_table_1'
    __table_args__ = {"schema": Config.POSTGRES_SCHEMA_DATA,
                      "extend_existing": True}

    id = Column(Integer,autoincrement=True, nullable=False, primary_key=True)
    name = Column(String(120))
    description = Column(String(120))
    value1= Column(Integer)
    #last_updated = Column('last_updated', DateTime, onupdate=datetime.datetime.now)

    def __init__(self, id, name, description, value1):
        self.id = id
        self.name = name
        self.description =description
        self.value1 = value1
        #self.last_updated = last_updated


class data_table_2(Base, UserMixin):

    __tablename__ = 'data_table_2'
    __table_args__ = {"schema": Config.POSTGRES_SCHEMA_DATA,
                      "extend_existing": True}

    id = Column(Integer, primary_key=True)
    name = Column(String(120), unique=True)
    description = Column(String(120))
    value1= Column(Integer)