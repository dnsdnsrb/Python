   Variable (5�H� cost/beta1_power  (5�{:	2_power  (5А�/_fc/beta1_power  (5�{:2_power  (5А�/ de0/fully_connected/biases� (�5��H�/Adam� �(�5�;�_1� �1(�5@�f[weights
�� �I(��T5�h,P/Adam
�� ��T(��T5M�2 _1
�� Բ�(��T57z��1/fully_connected/biases� ���(�5hwf�/Adam� ���(�5��ٳ_1� ܂�(�5���weights
�� ���(��5��  de1/fully_connected/weights/Adam
�� Н�(��5���� _1
�� ન(��5%�4_2/fully_connected/biases� �(�5bv�/Adam� ���(�5��&_1� �Ľ(�5߇ڈweights	1� �ʽ(��5ּ]Z/Adam	1� ���(��5�Ǆ* _1	1� ���(��5hd-b en1/fully_connected/biases� ���(�5���/Adam� ���(�5e~ �_1� ���(�5ʿ�
weights
�� ���(��T58dx�/Adam
�� ���(��T5]�e _1
�� ���(��T50�~2/fully_connected/biases� ���(�5.���/Adam� ̛�(�5@c�� !en2/fully_connected/biases/Adam_1� ܡ�(�5(f8weights
�� ��(��58$�/Adam
�� ���(��5��� _1
�� ���(��5�i�3/fully_connected/biases1 �ρ(�5� � /Adam1 �Ё(�5����_11 �ҁ(�5�o�weights	�1 �Ӂ(��5@�p�/Adam	�1 ���(��5\�� _1	�1 ���(��5��� fc1/Variable� �؈(�5�xX/Adam� �ވ(�5��_1� ��(�5X]L�var	1� ��(��5��E8/Adam	1� ؖ�(��5;�jI_1	1� �(��51��W fc2/Variable1 ��(�5���0/Adam1 ���(�5R_7{_11 ��(�50��var	�1 ��(��5Nϖ/Adam	�1 ԟ�(��56p�_1	�1 �˔(��5wj�
3/Variable
 ���((5X�F�/Adam
 ���((5���_1
 ���((5?ύ.var1
 ���(�59�g/Adam1
 ���(�5���0_11
 ���(�5X(O    �  A  w      r�        �� g �        ����                                  W���$uG�                                                                                                                                                                                                                                                                                                                                                                                                                                            t_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     print(ckpt.model_checkpoint_path)
    #     saver.restore(sess, ckpt.model_checkpoint_path)

    #train
    path = "AENN/varlist"
    print(path)
    saver = NNutils.save(path, sess)
    writer, merged = NNutils.graph(path, sess)

    test_indices = np.arange(len(teX))
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:128]

    st_time = datetime.now()
    print(st_time)
    step = 0
    for i in range(800):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trY), 128)): #trX[start] = [784, 0], trX[start:start]라 하면 [0, 784]로 변경됨. start~start까지의 갯수가 1개라서 0인 듯
            summary, \
            _, _,\
            loss_ae, loss_nn,\
            step = sess.run([merged,
                             trainop_ae, trainop_nn,
                             cost_ae, cost_nn,
                             global_step], feed_dict={X: trX[start:end], Y: trY[start:end]})

            if step % 50 == 0:
                print(step, datetime.now(), loss_nn, loss_ae)
                writer.add_summary(summary, step)

        accuracy, loss_ae, loss_nn = sess.run([acc, cost_ae, cost_nn], feed_dict={X: teX[test_indices], Y: teY[test_indices]})
        print(accuracy, loss_nn, loss_ae)

        saver.save(sess,  path + "/model.ckpt", step)

    end_time = datetime.now()
    print("학습 걸린 시간 = ", (end_time - st_time))