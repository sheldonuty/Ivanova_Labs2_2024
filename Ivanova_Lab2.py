# Задание 1. Есть два файла с данными турагенства: email.csv и username.csv.
# C ними нужно проделать все манипуляции, указанные в лекции 2, а именно:
# a) Группировка и агрегирование (сгруппировать набор данных по значениям в столбце, а затем вычислить среднее значение для каждой группы)

import pandas as pd

print("Task 1, a")
df_email = pd.read_csv('email.csv')
df_username = pd.read_csv('username.csv')
grouped_email = df_email.groupby('Trip count')['Total price'].mean()
print(grouped_email)

# b) Обработка отсутствующих данных (заполнение отсутствующих значений определенным значением или интерполяция отсутствующих значений)

print("Task 1, b")
df_email = df_email.fillna(value=0)
print(df_email)

df_username = df_username.fillna(value=0)
print(df_username)

# c) Слияние и объединение данных (соединить два DataFrames в определенном столбце)

print("Task 1, c")
merged_data = pd.merge(df_username,df_email,  on=['Identifier','First name','Last name','Trip count','Phone number'])
merged_data = merged_data.to_string()
print(merged_data)


# Задание 2. Преобразование данных (pivot):

# a) Нужно создать сводную таблицу так, чтобы в index были столбцы “Rep”, “Manager” и “Product”, а в values “Price” и “Quantity”.
# Также нужно использовать функцию aggfunc=[numpy.sum] и заполнить отсутствующие значения нулями.
# В итоге можно будет увидеть количество проданных продуктов и их стоимость, отсортированные по имени менеджеров и директоров.

import pandas as pd

print("Task 2, a")
df_sales = pd.read_csv('sales.csv')
pivot_table = pd.pivot_table(df_sales, index=['Rep', 'Manager', 'Product'],
                             values=['Price', 'Quantity'],
                             aggfunc="sum", fill_value=0) # При запуске отображается FutureWarning, где указано, что лучше использовать просто "sum"

pivot_table = pivot_table.sort_index()
print(pivot_table)

# b) Учебный файл (data.csv) + практика Dataframe.pivot. Поворот фрейма данных и суммирование повторяющихся значений.

import pandas as pd

print("Task 2, b")
df_data = pd.read_csv('data.csv')
pivot_table = pd.pivot_table(df_data, index='Date',
                                    columns='Product',
                                    values='Sales',
                                    aggfunc="sum", fill_value=0)
print(pivot_table)


# Задание 3. Визуализация данных (можно использовать любой из учебных csv-файлов).
#
# a) Необходимо создать простой линейный график из файла csv (два любых столбца, в которых есть зависимость)

import pandas as pd
import matplotlib.pyplot as plt

print("Task 3, a")
df_cars = pd.read_csv('cars.csv')
df_cars = df_cars.head(50)

df_cars.plot(kind = 'line', x = 'Horsepower', y ='Acceleration', color='g')
plt.xlabel('Мощность')
plt.ylabel('Ускорение')
plt.title('Зависимость ускорения от мощности')

plt.legend().remove()
plt.grid(True)
plt.show()

# b) Создание визуализации распределения набора данных. Создать произвольный датасет через np.random.normal или использовать датасет из csv-файлов, потом построить гистограмму.

import pandas as pd
import matplotlib.pyplot as plt

print("Task 3, b")
df_cars = pd.read_csv('cars.csv')
df_cars = df_cars.head(100)

plt.hist(df_cars['Horsepower'])
plt.xlabel('Мощность')
plt.ylabel('Частота')
plt.title('Распределение автомобилей по мощности')
plt.grid(True)
plt.show()


# c) Сравнение нескольких наборов данных на одном графике. Создать два набора данных с нормальным распределением или использовать данные из файлов.
# Оба датасета отразить на одной оси, добавить легенду и название.

import pandas as pd
import matplotlib.pyplot as plt

print("Task 3, c")
df_cars = pd.read_csv('cars.csv')
df_cars = df_cars.head(25)

plt.plot(df_cars['Horsepower'], df_cars['Acceleration'], 'b')
plt.plot(df_cars['Displacement'], df_cars['Acceleration'], 'r')

plt.title('Зависимость ускорения от мощности и расхода топлива автомобиля')
plt.xlabel('Мощность или Объем двигателя')
plt.ylabel('Ускорение')
plt.legend(['Horsepower vs. Acceleration','Displacement vs. Acceleration'])
plt.grid(True)

plt.show()

# d) Построение математической функции. Создать данные для x и y (где x это numpy.linspace, а y - заданная в условии варианта математическая функция).
# Добавить легенду и название графика.
# Функция sin

import numpy as np
import matplotlib.pyplot as plt

print("Task 3, d")
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.legend(['sin(x)'])
plt.title('График функции sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Функция cos

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.cos(x)

plt.plot(x, y)
plt.legend(['cos(x)'])
plt.title('График функции cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# e) Моделирование простой анимации. Создать данные для x и y (где x это numpy.linspace, а y - математическая функция). Запустить объект line, ввести функцию animate(i)
# c методом line.set_ydata() и создать анимированный объект FuncAnimation.
# a) Шаг 1: смоделировать график sin(x) (или cos(x)) в движении.
# b) Шаг 2: добавить к нему график cos(x) (или sin(x)) так, чтобы движение шло одновременно и оба графика отображались на одной оси.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("Task 3, e")
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
fig, ax = plt.subplots()

line_sin, = ax.plot(x, y_sin, color='b')
line_cos, = ax.plot(x, y_cos, color='red')
plt.legend(['sin(x)','cos(x)'])

def animate(i):
    line_sin.set_ydata(np.sin(x + i / 10))
    line_cos.set_ydata(np.cos(x + i / 10))
    return line_sin, line_cos


ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.show()


# Задание 4. Загрузка CSV-файла в DataFrame. Используя pandas, напишите скрипт, который загружает CSV-файл в DataFrame и отображает первые 5 строк df.

import pandas as pd

print("Task 4")
df = pd.read_csv('Climate.csv')
df = df.head(5)
df = df.to_string()
print(df)


# Задание 5. Выбор столбцов из DataFrame.
# a)	Используя pandas, напишите сценарий, который из DataFrame файла sales.csv выбирает только те строки, в которых Status = presented,
# и сортирует их по цене от меньшего к большему.

import pandas as pd

print("Task 5, a")
df_sales = pd.read_csv('sales.csv')

df_presented = df_sales[df_sales['Status'] == 'presented']
df_result = df_presented.sort_values('Price')
df_result = df_result.to_string()
print(df_result)

# b) Из файла climate.csv отображает в виде двух столбцов названия и коды (rw_country_code) тех стран, у которых cri_score больше 100, а fatalities_total не более 10.

import pandas as pd

print("Task 5, b")
df_climate = pd.read_csv('climate.csv')

df_filtered = df_climate[(df_climate['cri_score'] > 100) & (df_climate['fatalities_total'] <= 10)]
result = df_filtered[['rw_country_name', 'rw_country_code']]
result = result.to_string()
print(result)

# c) Из файла cars.csv отображает названия 50 первых американских машин, у которых расход топлива MPG не менее 25,
# а частное внутреннего объема (Displacement) и количества цилиндров не более 40. Названия машин нужно вывести в алфавитном порядке.

import pandas as pd

print("Task 5, c")
data = pd.read_csv('cars.csv')

filtered_data = data[(data['MPG'] >= 25) & (data['Displacement'] / data['Cylinders'] <= 40)]
final_cars = filtered_data['Car'].sort_values()
final_cars.head(50)
for car in final_cars:
    print(car)


# Задание 6.
# Вычисление статистики для массива numpy. Используя numpy, напишите скрипт, который загружает файл CSV в массив numpy и вычисляет среднее значение,
# стандартное отклонение и максимальное значение массива. Для тренировки используйте файл data.csv, а потом любой другой csv-файл от 20 строк.

import numpy as np

print("Task 6")
data = np.genfromtxt('data.csv', delimiter=',', names=True)

mean_data = np.mean(data['Sales'])
std_data = np.std(data['Sales'])
max_data = np.max(data['Sales'])

print("Среднее значение по dats.csv:", mean_data)
print("Стандартное отклонение по dats.csv:", std_data)
print("Максимальное значение по dats.csv:", max_data)

sales = np.genfromtxt('sales.csv', delimiter=',', names=True)

mean_sales = np.mean(sales['Price'])
std_sales = np.std(sales['Price'])
max_sales = np.max(sales['Price'])

print("\nСреднее значение по sales.csv:", mean_sales)
print("Стандартное отклонение по sales.csv:", std_sales)
print("Максимальное значение по sales.csv:", max_sales)


# Задание 7.
# Операции с матрицами: Используя numpy, напишите сценарий, который создает матрицу и выполняет основные математические операции,
# такие как сложение, вычитание, умножение и транспонирование матрицы.

import numpy as np

print("Task 7")
matrix_a = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

matrix_b = np.array([[10, 11, 12],
                     [13, 14, 15],
                     [16, 17, 18]])

matrix_sum = matrix_a + matrix_b
matrix_diff = matrix_a - matrix_b
matrix_dot = np.dot(matrix_a, matrix_b)
matrix_a_transpose = np.transpose(matrix_a)
matrix_b_transpose = np.transpose(matrix_b)

print("Матрица A:", matrix_a)
print("Матрица B:", matrix_b)
print("Сумма матриц:", matrix_sum)
print("Разность матриц:", matrix_diff)
print("Произведение матриц:", matrix_dot)
print("Транспонированная матрица A:", matrix_a_transpose)
print("Транспонированная матрица B:", matrix_b_transpose)