import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QGridLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QCursor
from PyQt5.QtCore import Qt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import logging
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体（SimHei），用于支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示

# 设置日志记录，用于调试和记录程序运行信息
logging.basicConfig(filename='digit_recognizer.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DigitRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        try:
            self.initUI()  # 初始化用户界面
            self.trainAndSaveModel()  # 训练模型并保存
            self.loadModel()  # 加载训练好的模型
        except Exception as e:
            logging.exception("初始化过程中发生错误")
            self.showErrorMessage("初始化错误", str(e))

    def initUI(self):
        """初始化用户界面"""
        self.setGeometry(300, 300, 900, 600)  # 设置窗口的位置和大小
        self.setWindowTitle('Lil_Rayyyの开源项目——手写数字识别')  # 设置窗口标题

        # 创建画板，背景色为黑色
        self.label = QLabel(self)
        canvas = QPixmap(280, 280)  # 创建一个280x280像素的画布
        canvas.fill(Qt.black)  # 将画布背景填充为黑色
        self.label.setPixmap(canvas)  # 将画布设置为标签的显示内容
        self.label.setCursor(QCursor(Qt.CrossCursor))  # 设置鼠标光标为十字形

        # 创建“识别”按钮，点击后将调用识别函数
        self.btn = QPushButton('识别', self)
        self.btn.clicked.connect(self.recognize)  # 连接按钮的点击事件到识别函数

        # 创建“清除”按钮，点击后将清空画板
        self.clear_btn = QPushButton('清除', self)
        self.clear_btn.clicked.connect(self.clear)  # 连接按钮的点击事件到清除函数

        # 创建显示识别结果的标签
        self.result_label = QLabel('识别结果：', self)

        # 创建显示模型训练结果的标签
        self.accuracy_label = QLabel(self)
        self.loss_label = QLabel(self)

        # 设置布局，将控件添加到网格布局中
        grid = QGridLayout()
        grid.addWidget(self.label, 0, 0, 1, 2)  # 将画板放在网格布局的第0行第0-1列
        grid.addWidget(self.btn, 1, 0)  # 将“识别”按钮放在第1行第0列
        grid.addWidget(self.clear_btn, 1, 1)  # 将“清除”按钮放在第1行第1列
        grid.addWidget(self.result_label, 2, 0, 1, 2)  # 将结果标签放在第2行第0-1列
        grid.addWidget(self.accuracy_label, 3, 0)  # 将准确率标签放在第3行第0列
        grid.addWidget(self.loss_label, 3, 1)  # 将损失标签放在第3行第1列

        self.setLayout(grid)  # 将网格布局设置为窗口的主要布局

        self.last_x, self.last_y = None, None  # 初始化鼠标位置，用于记录绘图时的上一个点

    def trainAndSaveModel(self):
        """训练并保存模型"""
        try:
            print("开始训练模型...")
            # 加载MNIST数据集，包含手写数字的训练和测试数据
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            # 数据预处理
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # 将训练数据重新调整形状并归一化
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # 将测试数据重新调整形状并归一化
            y_train = tf.keras.utils.to_categorical(y_train, 10)  # 将训练标签转换为one-hot编码
            y_test = tf.keras.utils.to_categorical(y_test, 10)  # 将测试标签转换为one-hot编码

            # 数据增强，增加训练数据的多样性
            datagen = ImageDataGenerator(
                rotation_range=15,  # 随机旋转图像
                width_shift_range=0.1,  # 随机水平平移图像
                height_shift_range=0.1,  # 随机垂直平移图像
                zoom_range=0.1  # 随机缩放图像
            )
            datagen.fit(x_train)  # 将数据增强应用于训练数据

            # 构建卷积神经网络模型
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),  # 第一个卷积层
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 最大池化层
                tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),  # 第二个卷积层
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 最大池化层
                tf.keras.layers.Flatten(),  # 将特征图展平成一维向量
                tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # 全连接层，带L2正则化
                tf.keras.layers.Dropout(0.5),  # Dropout层，防止过拟合
                tf.keras.layers.Dense(10, activation='softmax')  # 输出层，10个类别的softmax激活函数
            ])

            # 编译模型，指定优化器、损失函数和评价指标
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adam优化器，学习率0.001
                               loss='categorical_crossentropy',  # 使用交叉熵作为损失函数
                               metrics=['accuracy'])  # 评价指标为准确率

            # 使用早停法防止过拟合
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # 训练模型，使用数据增强
            history = self.model.fit(datagen.flow(x_train, y_train, batch_size=128),
                                     epochs=50,  # 训练50个周期
                                     validation_data=(x_test, y_test),  # 使用测试数据作为验证集
                                     callbacks=[early_stopping])  # 使用早停回调函数

            # 保存模型到文件
            self.model.save('mnist_model.h5')
            print("模型训练完成并已保存。")

            # 绘制训练和验证的损失和准确率
            self.plotTrainingHistory(history)
        except Exception as e:
            logging.exception("模型训练失败")  # 记录模型训练失败的信息
            self.showErrorMessage("模型训练错误", str(e))  # 显示错误信息

    def plotTrainingHistory(self, history):
        """绘制训练和验证的损失和准确率，并显示在界面上"""
        # 绘制准确率图
        plt.figure(figsize=(4.5, 3.5))  # 设置图表大小
        plt.plot(history.history['accuracy'], label='训练准确率')  # 绘制训练准确率曲线
        plt.plot(history.history['val_accuracy'], label='验证准确率')  # 绘制验证准确率曲线
        plt.xlabel('Epoch')  # 设置X轴标签为"Epoch"
        plt.ylabel('准确率')  # 设置Y轴标签为"准确率"
        plt.legend()  # 显示图例
        plt.title('准确率')  # 设置图表标题为"准确率"

        buf = BytesIO()  # 创建一个字节缓冲区，用于保存图像
        plt.savefig(buf, format='png')  # 将图表保存为PNG格式到缓冲区
        buf.seek(0)  # 将缓冲区指针重置为起始位置
        img = Image.open(buf)  # 打开缓冲区中的图像
        img = img.resize((400, 300), Image.Resampling.LANCZOS)  # 调整图像大小
        img = img.convert("RGB")  # 将图像转换为RGB模式
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)  # 将PIL图像转换为QImage
        self.accuracy_label.setPixmap(QPixmap.fromImage(qimg))  # 将QImage设置为标签的显示内容
        buf.close()  # 关闭缓冲区
        plt.close()  # 关闭图表窗口

        # 绘制损失图
        plt.figure(figsize=(4.5, 3.5))
        plt.plot(history.history['loss'], label='训练损失')  # 绘制训练损失曲线
        plt.plot(history.history['val_loss'], label='验证损失')  # 绘制验证损失曲线
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.title('损失')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        self.loss_label.setPixmap(QPixmap.fromImage(qimg))
        buf.close()
        plt.close()

    def loadModel(self):
        """加载预训练模型"""
        try:
            self.model = tf.keras.models.load_model('mnist_model.h5')  # 从文件加载模型
            logging.info("模型加载完成。")  # 记录模型加载完成信息
        except Exception as e:
            logging.exception("模型加载失败")  # 记录模型加载失败信息
            self.showErrorMessage("模型加载错误", str(e))  # 显示错误信息

    def preprocess_image(self, img):
        """图像预处理步骤"""
        # 调整图像大小为28x28像素
        img = tf.image.resize(img, (28, 28))
        # 将像素值归一化到0到1之间
        img = img / 255.0
        # 不再反转颜色，保持白色数字和黑色背景不变
        # 确保图像为三维张量 (height, width, channels)
        if img.ndim == 2:  # 如果图像是二维的，添加通道维度
            img = np.expand_dims(img, axis=-1)
        return img

    def recognize(self):
        """识别画板中的数字"""
        try:
            logging.debug("开始识别过程")  # 记录识别过程的开始

            # 获取画板图像
            img = self.label.pixmap().toImage()  # 将标签中的pixmap转换为QImage
            img = img.convertToFormat(QImage.Format_Grayscale8)  # 转换为灰度图像格式
            width = img.width()  # 获取图像宽度
            height = img.height()  # 获取图像高度

            # 将图像数据转换为 NumPy 数组
            ptr = img.bits()  # 获取图像的字节数据
            ptr.setsize(height * width)  # 设置字节数据的大小
            img_arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 1))  # 将字节数据转换为三维数组 (height, width, channels)

            logging.debug(f"原始图像大小: {img_arr.shape}")  # 记录原始图像的大小

            # 预处理图像
            img_arr = self.preprocess_image(img_arr)  # 对图像进行预处理

            # 确保图像形状为 (28, 28, 1)
            if img_arr.shape != (28, 28, 1):
                raise ValueError(f"图像尺寸不匹配：期望 (28, 28, 1)，但得到 {img_arr.shape}")  # 如果图像形状不匹配，则抛出异常

            # 添加批次维度
            input_data = np.expand_dims(img_arr, axis=0)  # 在第0轴添加一个批次维度
            logging.debug(f"输入数据形状: {input_data.shape}")  # 记录输入数据的形状

            # 预测
            prediction = self.model.predict(input_data)  # 使用模型预测输入数据
            predicted_digit = np.argmax(prediction, axis=1)[0]  # 获取预测结果中概率最大的类别
            logging.debug(f"识别结果: {predicted_digit}")  # 记录识别结果

            # 显示结果
            self.result_label.setText(f'识别结果：{predicted_digit}')  # 将识别结果显示在标签上
        except Exception as e:
            logging.exception("识别过程出现错误")  # 记录识别过程中出现的错误
            self.showErrorMessage("识别错误", str(e))  # 显示错误信息

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.LeftButton:  # 检查鼠标左键是否被按下
            self.last_x, self.last_y = event.x(), event.y()  # 记录鼠标按下时的位置

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.last_x is None:  # 如果鼠标没有按下
            return  # 直接返回，不做任何操作

        # 创建QPainter对象，用于在画布上绘图
        painter = QPainter(self.label.pixmap())
        painter.setPen(QPen(Qt.white, 12, Qt.SolidLine))  # 设置画笔颜色为白色，线条宽度为12像素
        painter.drawLine(self.last_x - self.label.pos().x(), self.last_y - self.label.pos().y(),
                         event.x() - self.label.pos().x(), event.y() - self.label.pos().y())  # 绘制一条从上一个点到当前点的直线
        painter.end()  # 结束绘图
        self.update()  # 更新界面

        self.last_x, self.last_y = event.x(), event.y()  # 更新上一个点的位置

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        self.last_x = None  # 重置上一个点的位置
        self.last_y = None  # 重置上一个点的位置

    def clear(self):
        """清空画板，但保留准确率图像"""
        canvas = QPixmap(280, 280)  # 创建一个新的画布
        canvas.fill(Qt.black)  # 将画布背景填充为黑色
        self.label.setPixmap(canvas)  # 将新的画布设置为标签的显示内容
        self.result_label.setText('识别结果：')  # 重置识别结果标签的文本

    def showErrorMessage(self, title, message):
        """显示错误信息"""
        QMessageBox.critical(self, title, message)  # 显示一个错误消息框，标题为title，内容为message


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    recognizer = DigitRecognizer()  # 创建DigitRecognizer窗口对象
    recognizer.show()  # 显示窗口
    sys.exit(app.exec_())  # 进入应用程序的主循环