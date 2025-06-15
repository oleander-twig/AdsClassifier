import logging 
import mysql.connector

logging.basicConfig(level=logging.INFO)

config = {
  'user': 'scott',
  'password': 'password',
  'host': '127.0.0.1',
  'database': 'employees',
  'raise_on_warnings': True
}

def data_load() -> None:

    cnx = mysql.connector.connect(**config)

    if cnx and cnx.is_connected():

        with cnx.cursor() as cursor:

            result = cursor.execute("SELECT * FROM actor LIMIT 5")

            rows = cursor.fetchall()

            for rows in rows:

                print(rows)

        cnx.close()

    else:

        print("Could not connect")

    cnx.close()

    return 

if __name__ == "__main__":
    data_load()




