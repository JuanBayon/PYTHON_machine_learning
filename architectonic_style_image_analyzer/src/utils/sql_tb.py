from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

class MySQL:

    def __init__(self, IP_DNS, USER, PASSWORD, BD_NAME, PORT):
        self.IP_DNS = IP_DNS
        self.USER = USER
        self.PASSWORD = PASSWORD
        self.BD_NAME = BD_NAME
        self.PORT = PORT
        self.SQL_ALCHEMY = 'mysql+pymysql://' + self.USER + ':' + self.PASSWORD + '@' + self.IP_DNS + ':' + str(self.PORT) + '/' + self.BD_NAME


    def create_engine(self):
        engine = create_engine(self.SQL_ALCHEMY, poolclass=NullPool)
        conn = engine.connect()
        print("Connected to MySQL server [" + self.BD_NAME + "]")
        return conn, engine


    def execute_engine(self, conn, sql, args=None):
        result = 0
        if args:
            try:
                result = conn.execute(sql, args).fetchall()
                print("Executed \n\n" + str(sql) + "\n\n successfully")
            except Exception as error:
                print(error)
        else:
            try:
                result = conn.execute(sql).fetchall()
                print("Executed \n\n" + str(sql) + "\n\n successfully")
            except Exception as error:
                print(error)
        return result


    def close_engine(self, conn, engine):
        conn.close()
        engine.dispose()
        print("Close connection with MySQL server [" + self.BD_NAME + "]")