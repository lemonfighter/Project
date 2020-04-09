import sqlite3


class SQL3Controllor:
    def __init__(self, db_path: str):
        print("===== Connect to db =====")
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def __del__(self):
        self.conn.close()

    def cmd(self, cmd: str):
        self.conn.execute(cmd)

    def create_table(self, table_name: str, row_detail: str):
        print('=== Create New Table ===')
        cmd = ''
        self.conn.execute(cmd + 'CREATE TABLE ' + table_name + ' (''' + row_detail + ''');''')
        self.conn.commit()

    def drop_table(self, table_name: str):
        self.conn.execute('DROP TABLE ' + table_name)
        self.conn.commit()

    def get_data(self, table_name: str, to_list=True, condition=None):
        cmd = "SELECT * from " + table_name
        cmd += " where " + condition if condition is not None else ""
        data = self.conn.execute("SELECT * from " + table_name)
        if to_list:
            return [row for row in data]
        return data

    def insert_data(self, table_name: str, row_name_list: list, row_data_list: list):
        cmd = "INSERT INTO " + table_name + " ("
        for n in row_name_list:
            cmd += "{},".format(str(n))
        cmd = cmd[:-1] + ") VALUES ("
        for d in row_data_list:
            if isinstance(d, str):
                cmd += "'{}',".format(d)
            else:
                cmd += "{},".format(str(d))
        cmd = cmd[:-1] + ")"
        self.conn.execute(cmd)
        self.conn.commit()

    def update_data(self, table_name: str, row_name_list: list, row_data_list: list, condition: str):
        cmd = "UPDATE " + table_name + " SET "
        for n, d in zip(row_name_list, row_data_list):
            cmd += "{}=".format(str(n))
            cmd += "'{},'".format(d) if isinstance(d, str) else "{},".format(str(d))
        cmd = cmd[:-1] + " WHERE " + condition
        self.conn.execute(cmd)
        self.conn.commit()

    def delete_data(self, table_name: str, condition: str):
        self.conn.execute("DELETE FROM " + table_name + " WHERE " + condition)
        self.conn.commit()


if __name__ == '__main__':
    sql = SQL3Controllor('a.db')
    print(sql.get_data(table_name='Test'))
    sql.update_data('Test', ["Text"], ["'HIIIIII'"], "ID=1")
    print(sql.get_data(table_name='Test'))

