from pyqpanda import *

if __name__ == "__main__":
    machine = init_quantum_machine(QMachineType.CPU)
    q = machine.qAlloc_many(2)
    x = var(1)
    y = var(2)

    # 变分电路部分
    vqc = VariationalQuantumCircuit()
    vqc.insert(VariationalQuantumGate_H(q[0]))
    vqc.insert(VariationalQuantumGate_RX(q[0], x))
    vqc.insert(VariationalQuantumGate_RY(q[1], y))
    vqc.insert(RZ(q[0], 3))

    # 常规电路部分
    circ = QCircuit()
    circ.insert(RX(q[0],3))
    circ.insert(RY(q[1],4))

    # 变分电路 + 常规电路 --> 送入 量子程序 QProg
    vqc.insert(circ)
    circuit1 = vqc.feed()  # 量子程序 QProg 无法直接加载可变量子线路，但是我们可以通过调用可变量子线路的 feed 接口来生成一个普通量子线路。
    prog = QProg()
    prog.insert(circuit1)
    print("prog", prog)
    print(convert_qprog_to_originir(prog, machine))

    # 改变 变分电路 里的变量  -->重新 送入 量子程序 QProg
    x.set_value([[3.]])
    y.set_value([[4.]])
    circuit2 = vqc.feed()
    prog2 = QProg()
    prog2.insert(circuit2)
    print(convert_qprog_to_originir(prog2, machine))
    print("prog2", prog2)