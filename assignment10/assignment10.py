# Сам же массив это простая структура, для которой мы все подгтовили

from queue import Queue
import numpy as np

class SystolicArrayCell:
    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0

        # частичная сумма: передается вертикально
        self.partial_sum = 0
        self.partial_sum_out = 0
        # активации: передается горизонтально
        self.activation = 0
        self.activation_out = 0

        # веса - w 
        self.weight = 0
        
        # а это нам нужно, чтобы принимать данные от соседних ячеек  
        self.input_activation = None
        self.input_partial_sum = None

    def set_weight(self, weight):
        self.weight = weight

    def connect(self, pos_x, pos_y, array):
        self.pos_x = pos_x
        self.pos_y = pos_y

        # Connect to the left neighbor if not the first column
        if self.pos_x == 0:
            self.input_activation = array.input[self.pos_y]
        else:
            self.input_activation = array.cells[self.pos_y][self.pos_x - 1]

        # Connect to the top neighbor if not the first row
        if self.pos_y == 0:
            self.input_partial_sum = None
        else:
            self.input_partial_sum = array.cells[self.pos_y - 1][self.pos_x]

    def read(self):
        # считайте данные соседа слева, помните что у слева сосед может отсутствовать
        if type(self.input_activation) is Queue:
            if self.input_activation.empty():
                self.activation = 0
            else:
                self.activation = self.input_activation.get()
        else:
            # а если он есть, то просто вохьмите те данные которые пришли от него
            self.activation = self.input_activation.activation_out

        # Аналогично поступите и с верхним соседом.
        if self.input_partial_sum is None:
            self.partial_sum = 0
        else:
            self.partial_sum = self.input_partial_sum.partial_sum_out


    def compute(self):
        # Perform computation
        self.partial_sum += self.activation * self.weight
        # Update outputs
        self.partial_sum_out = self.partial_sum
        self.activation_out = self.activation


class SystolicArray:
    # Делаем квадратный массив
    def __init__(self, array_size):
        self.array_size = array_size

        # наш массив ячеек
        self.cells = []
        for _ in range(self.array_size):
            row = []
            for _ in range(self.array_size):
                cell = SystolicArrayCell()
                row.append(cell)
            self.cells.append(row)

        # В качестве входов и выходов будет наша очередь
        self.input = [Queue() for _ in range(self.array_size)]
        self.output = [Queue() for _ in range(self.array_size)]

        # не забудем связать сетки между собой
        for row_num, row in enumerate(self.cells):
            for col_num, cell in enumerate(row):
                cell.connect(col_num, row_num, self)

    # Заполним веса. Веса мы заполняем напрямую, так в реальности не делается
    def fill_weights(self, weights):
        for row_num, row in enumerate(weights):
            for col_num, weight in enumerate(row):
                self.cells[row_num][col_num].set_weight(weight)

    def fill_activations(self, activations):
        # надо западдить активации нулями в виде нижнего треугольника
        for row_num in range(self.array_size):
            for _ in range(row_num):
                self.input[row_num].put(0)

        # Еще надо выполнить транспонирование чтобы перемножение корректно работало
        for row_num in range(self.array_size):
            col = [activations[x][row_num] for x in range(self.array_size)]
            for activation in col:
                self.input[row_num].put(activation)
    
    def read(self):
        for row in self.cells:
            for cell in row:
                cell.read()

    def compute(self):
        for row in self.cells:
            for cell in row:
                cell.compute()
        # переносим данные 
        for col_num in range(self.array_size):
            self.output[col_num].put(self.cells[-1][col_num].partial_sum_out)

    # каждый такт состоит из read и compute
    def cycle(self):
        self.read()
        self.compute()

    # вычисляем
    def run(self):
        # Почему такое количество тактов нам потребуется?
        for _ in range(3*self.array_size - 2):
            self.cycle()

        return self.get_outputs()

    # Забираем выходы
    def get_outputs(self):
        ret = []

        for col_num in range(self.array_size):
            for _ in range(col_num + self.array_size - 1):
                self.output[col_num].get()

        # Транспонируем результат
        for row_num in range(self.array_size):
            row = []
            for output_col in self.output:
                row.append(output_col.get())
            ret.append(row)

        return ret
    

# Проверочка
myArray = SystolicArray(3)

activations = np.array([[1,2,3],[4,5,6],[7,8,9]]).astype(np.float32)
weights = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)

#activations = np.random.randint(low=-5, high=5, size=(3, 3))
myArray.fill_activations(activations)

#weights = np.random.randint(low=-5, high=5, size=(3, 3))
myArray.fill_weights(weights)

res = myArray.run()
print(res)
print(np.matmul(activations, weights))
assert (res == np.matmul(activations, weights)).all()
print('It\'s ok.')