import os


def main():
    dir_in = "/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_AMFM/ASPEC/CJ"
    dir_out = "/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_AMFM/ASPEC/CJ/Processed"

    # metodo que sirve para generar de los ficheros que tenemos las referencias y submisisons para pruebas posteriores.

    if not os.path.exists(dir_out):
        print("...creating " + dir_out)
        os.makedirs(dir_out)

    my_list = os.listdir(dir_in)

    for file in my_list:
        filename_in = dir_in + '/' +file
        filename_out = dir_out + '/'+ file

    with open(filename_in, 'r') as f_in, open(filename_out, 'w', encoding='utf-8') as f_out:

        lines = f_in.readlines()
        count = 0
        for line in lines:
            count = count + 1
            if (1==count or 7 == count or count == 13  or count == 19 or  count == 25):
                linesplit = line.split("-")
                Subtask = linesplit[2].split(".")
                f_out.write(Subtask[0] + '\t')
            if (count == 4 or count == 10 or count == 16 or count == 22 or count == 28):
                spacesplit = line.split(" ")
                result = spacesplit[4]
                f_out.write(result )

            # # f_in.readlines()
            # new_line = line.split("\t")
            # # finalline= new_line[1].split("。")
            # # f_out.write(new_line[0] + '\n'+new_line[1]+'\n'+new_line[2]+'\n')
            # f_ref_out.write(new_line[1] + '\n')
            # f_sub_out.write(new_line[2] + '\n')


if __name__ == '__main__':
    main()
