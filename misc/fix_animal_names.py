import os
import re
import sys



if __name__ =="__main__":

    base_path = sys.argv[1]
    s_name = sys.argv[2]
    change_to = sys.argv[3]
    f_type = sys.argv[4]
    print 
    for f1 in os.listdir(base_path):

        if os.path.isdir(os.path.join(base_path,f1)):

            for f2 in os.listdir(os.path.join(base_path,f1)):

                full_path = os.path.join(base_path,f1,f2)
                
                new_path = os.path.join(base_path,f1,f2.replace(s_name,change_to))
                os.rename(full_path,new_path)
        else:

            full_path = os.path.join(base_path,f1)
            new_path = os.path.join(base_path,f1.replace(s_name,change_to))
            os.rename(full_path,new_path)


