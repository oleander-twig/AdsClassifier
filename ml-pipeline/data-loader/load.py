import logging 
import mysql.connector
from config import create_config, Config
import csv

logging.basicConfig(level=logging.INFO)

def create_fake_data(cfg: Config):
    cnx = mysql.connector.connect(host=cfg.database.host,
                                user=cfg.database.user,
                                password=cfg.database.password,
                                database=cfg.database.name)
    if cnx and cnx.is_connected():
        with cnx.cursor() as cursor:
            cursor.execute("""
                DROP DATABASE IF EXISTS mydb;
            """)

            cursor.execute("""
                CREATE DATABASE IF NOT EXISTS mydb;
            """)

            cursor.execute("""
                DROP TABLE IF EXISTS mydb.tmp;
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mydb.tmp (
                    Id INT NOT NULL AUTO_INCREMENT,
                    text VARCHAR(255),
                    category VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (Id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            cursor.execute("""
                INSERT INTO mydb.tmp
                (Id, `text`, category, created_at)
                VALUES(0, 'Курортная сеть «Азовский» - отели Азовский и Азовленд для семейного отдыха на Азовском море в Крыму', 'Туризм, спорт и отдых', CURRENT_TIMESTAMP);
            """)

            cursor.execute("""
                INSERT INTO mydb.tmp
                (Id, `text`, category, created_at)
                VALUES(0, 'Недорогие окна ПВХ в Минске по низкой цене от производителя - скидки на пластиковые окна и стеклопакеты | STROY-INFO', 'Строительные, отделочные материалы, сантехника', CURRENT_TIMESTAMP);
            """)

            cursor.execute("""
                INSERT INTO mydb.tmp
                (Id, `text`, category, created_at)
                VALUES(0, 'Официальный магазин Nivona Кофемашины Nivona – Официальный сайт производителя на территории России.', 'Бытовая техника', CURRENT_TIMESTAMP);
            """)

            cursor.execute("""
                INSERT INTO mydb.tmp
                (Id, `text`, category, created_at)
                VALUES(0, 'Студия событий Агентсити', 'Массовые мероприятия', CURRENT_TIMESTAMP);
            """)

            cnx.commit()
    cnx.close()


def data_load() -> None:

    cfg = create_config()

    if cfg.mode:
        create_fake_data(cfg=cfg)

    cnx = mysql.connector.connect(host=cfg.database.host,
                                  user=cfg.database.user,
                                  password=cfg.database.password,
                                  database=cfg.database.name)

    if cnx and cnx.is_connected():
        with cnx.cursor() as cursor:
            result = cursor.execute("SELECT text, category FROM mydb.tmp WHERE DATE(created_at) = CURDATE()")
            rows = cursor.fetchall()

    else:
        print("Could not connect")

    cnx.close()

    # with open('output.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(rows)

    return 

if __name__ == "__main__":
    data_load()




