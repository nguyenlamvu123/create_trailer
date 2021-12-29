import pandas as pd

class Baitapbuoi4:
    
    danhsach = []
    def __init__(self,
##                 danhsach = [],
##                 lissst = [],
##                 nhhhan = [''],
                 ):
##        self.danhsach = danhsach 
##        self.lissst = lissst
##        self.nhhhan = nhhhan
        pass 

    def Ex2_createSeries(
        self,
        lissst = [],
##        nhhhan = [''],
        ):
        return self.danhsach.append(
            pd.Series(
                lissst,
##                self.lissst,
##                index = [],
                )
            )

    def Ex2_createdf(self):
        return pd.DataFrame(
            self.danhsach,
##            columns=[],
##            index=["first_name", "last_name", "age", "job", "country"],
            )
    
    def Ex2_csv(
        self,
        ):
        return self.Ex2_createdf().to_csv('out.csv')

    def Ex3(self):
        df = pd.read_csv('out.csv')
        df.loc[0] = df.loc[0].astype(str)+ '#'#"my_first_name"
        df.loc[1] = df.loc[1].astype(str)+ '@'#"my_last_name"
        print('&&&&&&&&&&', df.to_string())
        return self.Ex4(df)

    def Ex4(
        self,
        df
        ):
##        df = self.Ex3()
        print(
            'all rows whose "country" is "Vietnam"', 
            df.loc[
                :,
                df.loc[4]=="Vietnam"
                ],
            )
        print(
            'all rows whose "country" is not "Vietnam"',
            df.loc[
                :,
                df.loc[4]!="Vietnam"
                ],
            )
        print(
            'all rows whose "age" is lower than 20',
            df.loc[
                :,
                df.loc[2].astype(int)<20
                ],
            )
        
##    for s in danhsach:
##        print(s)            
s1 = []
for zs in ["first_name", "last_name", "age", "job", "country"]:
    ss = [zs+ str(i) for i in range(1,11) if zs != "age"]
    if zs == "country":
        ss[3] = "Vietnam"
    elif zs == "age":
        ss = [20+ ii for ii in range(-12, 38, 5)]
    s1.append(ss)#;print(ss)

s = Baitapbuoi4()
for ss in s1:
    s.Ex2_createSeries(
        lissst = ss,
        )
##for ss in s.danhsach:
##    print(ss)
####print(s.Ex2_createdf().to_string())
####s.Ex2_csv()
####s.Ex3()
