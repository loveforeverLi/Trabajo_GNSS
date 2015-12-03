
import numpy as np
import gpstk
import matplotlib.pyplot as plt

def getdata(nfile,ofile,strsat=None): #one week data
    f1,f2=gpstk.L1_FREQ_GPS,gpstk.L2_FREQ_GPS
    alfa=1.0/((f1**2/f2**2)-1)
    t=[] #observation epoch on code and phase 
    Iphase,Icode,IPPS={},{},{} #dicc: key=time observer; values=iono delay and IPP
    VTECphase,VTECcode,ELEV={},{},{}
    PhaseL1,PhaseL2={},{}
    CodeL1,CodeL2={},{}
    
    oheader,odata=gpstk.readRinex3Obs(ofile,strict=True) 
    nheader,ndata=gpstk.readRinex3Nav(nfile)
    
    bcestore = gpstk.GPSEphemerisStore() 
    
    for ndato in ndata:
        ephem = ndato.toGPSEphemeris()
        bcestore.addEphemeris(ephem)
    bcestore.SearchNear() 
   
    
    for observation in odata:
        sats=[satID for satID, datumList in observation.obs.iteritems() if str(satID).split()[0]=="GPS" ] 
        obs_types = np.array([i for i in oheader.R2ObsTypes])
        if 'C1' and 'P2' and 'L1' and 'L2' in obs_types:
            for sat in sats:
                if  str(sat)==strsat :#Return for a specific satellite
                    eph = bcestore.findEphemeris(sat, observation.time) 
                    Tgd=eph.Tgd
                    sat_pos = eph.svXvt(observation.time)
                    rec_pos = gpstk.Position(oheader.antennaPosition[0], oheader.antennaPosition[1], oheader.antennaPosition[2]).asECEF()
                    elev = oheader.antennaPosition.elvAngle(sat_pos.x)
                    azim = oheader.antennaPosition.azAngle(sat_pos.x)
                    time=observation.time
                    R=6.378e6 #earth radius
                    mapp=1/np.cos(np.arcsin(R/(R+350000))*np.sin(elev))
                    IPP=rec_pos.getIonosphericPiercePoint(elev, azim, 350000).asECEF()
                    t.append(np.trunc(gpstk.YDSTime(time).sod))
                   
                    if np.size(np.where(obs_types=='C1'))>0 and np.size(np.where(obs_types=='P2'))>0 and np.size(np.where(obs_types=='L1'))>0 and np.size(np.where(obs_types=='L2'))>0: 
                        
                        C1_idx = np.where(obs_types=='C1')[0][0] 
                        P2_idx = np.where(obs_types=='P2')[0][0]
                        R1=observation.getObs(sat, C1_idx).data 
                        R2=observation.getObs(sat, P2_idx).data
                        
                        L1_idx = np.where(obs_types=='L1')[0][0]
                        L2_idx = np.where(obs_types=='L2')[0][0]
                        L1=observation.getObs(sat, L1_idx).data*gpstk.L1_WAVELENGTH_GPS 
                        L2=observation.getObs(sat, L2_idx).data*gpstk.L2_WAVELENGTH_GPS
                       
                        if R2<3e7 and R1<3e7 and L2<3e7 and L1<3e7: #Distances should be in order of 1e7 meters, more than that is considered an error  
                            
                            iono_delay_c=alfa*(R2-R1) 
                            iono_delay_p=alfa*(L1-L2)
                            
                            vtec_C=iono_delay_c/mapp
                            vtec_P=iono_delay_p/mapp
                            
                            VTECcode[np.trunc(gpstk.YDSTime(time).sod)]=vtec_C
                            VTECphase[np.trunc(gpstk.YDSTime(time).sod)]=vtec_P
                            Icode[np.trunc(gpstk.YDSTime(time).sod)]=iono_delay_c
                            Iphase[np.trunc(gpstk.YDSTime(time).sod)]=iono_delay_p
                            ELEV[np.trunc(gpstk.YDSTime(time).sod)]=elev
                            IPPS[np.trunc(gpstk.YDSTime(time).sod)]=IPP
                            PhaseL1[np.trunc(gpstk.YDSTime(time).sod)]=L1
                            PhaseL2[np.trunc(gpstk.YDSTime(time).sod)]=L2
                            CodeL1[np.trunc(gpstk.YDSTime(time).sod)]=R1
                            CodeL2[np.trunc(gpstk.YDSTime(time).sod)]=R2
                            
                            #stec=(iono_delay_p*f1**2)/(-40.3) #STEC delay on phase [mm]
                            #vtec=stec/mapp #vertical delay!
        else:
            print "Needs both L1 and L2 frequencies to compute delay"
            break
    
    
    return t,Icode,Iphase,VTECphase,ELEV,IPPS,PhaseL1,PhaseL2,CodeL1,CodeL2,Tgd




def getdata_stationpair(station1,station2,strsat=None): 
    s1n_file,s1o_file=station1[0],station1[1] #station pair
    s2n_file,s2o_file=station2[0],station2[1] 
    
    t1,Icode1,Iphase1,VTECphase1,ELEV1,IPP1,L11,L12,C11,C12,Tgd=getdata(s1n_file,s1o_file,strsat)
    t2,Icode2,Iphase2,VTECphase2,ELEV2,IPP2,L21,L22,C21,C22,__=getdata(s2n_file,s2o_file,strsat)
    
    return t1,t2,Icode1,Iphase1,Icode2,Iphase2,VTECphase1,VTECphase2,ELEV1,ELEV2,IPP1,IPP2,L11,L12,L21,L22,C11,C12,C21,C22,Tgd 



def get_arcs(t,Icode,Iphase,ELEV,IPPS,L1,L2,C1,C2): 
    #Returns arcs with observations time, phase & code delay,IPP
    #For every station returns a dictionary
    ##key: Number of arc observed in that day (starting with 0).
    ##values: Phase, Code, etc in that arc
    Phase,Code,Elevation=[],[],[]
    PhaseL1,PhaseL2=[],[]
    CodeL1,CodeL2,IPP=[],[],[]
    notfound=0 
    tnew=[]
    for i in t:
        if i in Iphase.keys(): #all times in 't' should correspond to the keys in the dictionaries.
            if ELEV[i]>10:
                Phase.append(Iphase[i])
                Code.append(Icode[i])
                Elevation.append(ELEV[i])
                PhaseL1.append(L1[i])
                PhaseL2.append(L2[i])
                CodeL1.append(C1[i])
                CodeL2.append(C2[i])
                IPP.append(IPPS[i])
                tnew.append(i)
        else:
            #print "Tiempo no encontrado: ",i
            notfound+=1
            
    if notfound>0:
        print "Tiempos no encontrados :",notfound
    
    t=adjust_times(tnew)
    
    Phase,Code,t,Elevation,PhaseL1,PhaseL2,CodeL1,CodeL2=np.array(Phase),np.array(Code),np.array(t),np.array(Elevation),np.array(PhaseL1),np.array(PhaseL2),np.array(CodeL1),np.array(CodeL2)
    
    limits=[]
    for i in range(1,len(t)):
        if t[i]-t[i-1]>3600: 
            limits.append(i)

    arcs=np.split(range(t.size),limits)
    obs={} #Key: Number arc, Values: array time of observations , arrayPhase Delays, array code etc.
    i=0
    for arc in arcs:
        obs[i]=[t[arc],Phase[arc],Code[arc],Elevation[arc],PhaseL1[arc],PhaseL2[arc],CodeL1[arc],CodeL2[arc]]
        i+=1
    return obs


def match_arcs(obs1,obs2):#match arcs on the time of the day
    if len(obs1)==1:
        if obs1[0][0][0]/3600>12:
            obs1[1]=obs1[0]
            del(obs2[0])
    if len(obs2)==1:
        if obs2[0][0][0]/3600>12:
            obs2[1]=obs2[0]
            del(obs2[0])
    return obs1,obs2


def adjust_times(t): #observation's times should be a multiple of 30.  
    np.array(t)
    for i in range(len(t)):
        if t[i]%30!=0:
            t[i]=t[i]+(30-t[i]%30)
    return t


def get_indexes(time1,time2):
    tboth=np.intersect1d(time1,time2)
    #print "\n# Times in common: ",tboth.size
    t1index=np.array([i for i in range(time1.size) if time1[i] in tboth ])
    t2index=np.array([i for i in range(time2.size) if time2[i] in tboth ])
    return t1index,t2index


def getIPPS(IPP,tboth): #returns IPP of observations in a time
    DIPP={}
    for t in IPP:
        if t%30!=0:
            tn=t+(30-t%30)
            if tn in tboth:
                DIPP[tn]=IPP[t]
        else:
            if t in tboth:
                DIPP[t]=IPP[t]
    return DIPP


def distance_betweenIPPS(IPP1,IPP2,tboth): #Returns distance between IPP's
    DIPP1=getIPPS(IPP1,tboth)
    DIPP2=getIPPS(IPP2,tboth)
    D=[]
    for i in tboth:
        d=np.sqrt((DIPP1[i][0]-DIPP2[i][0])**2+(DIPP1[i][1]-DIPP2[i][1])**2+(DIPP1[i][2]-DIPP2[i][2])**2)
        D.append(d)
    D=np.array(D)
    return D


def datajump(lI,threshold=0.5): #Input: lI=L1-L2, times. Detects jumps in data depending on a threshold
    jumps=[]
    jumps=np.where(np.abs(np.diff(np.hstack(([0],lI))))>threshold) 
    return jumps[0] 


def sub_arcs(lI,jumps): #returns intervals of each sub-arc inside an arc
    miniarcs=np.split(range(lI.size),jumps)
    return miniarcs


def remove_short(miniarcs1,miniarcs2): #Deletes short arcs, on a subarc
    toerase=[]
    for arc in range(len(miniarcs1)):
        #print "SubArc # ",arc," Has ",len(miniarcs1[arc])," points" 
        #gets sub-arcs with les than 10 elements
        if miniarcs1[arc].size<10:
            toerase.append(arc)
            #print "Sub-Arcs deleted",arc,miniarcs1[arc]
    elim={}
    for j in range(len(miniarcs2)):#eah sub arc
        for k in range(miniarcs2[j].size):
            for arc in toerase: #for every arc to delete
                if miniarcs2[j][k] in miniarcs1[arc]:
                    if j in elim:
                        elim[j].append(k)
                    else:
                        elim[j]=[]
                        elim[j].append(k) #sub arc number in miniarcs2 and where is in that sub-arc
    #print "Indices de subarcos a eliminar: ",toerase       
    #for arc in toerase:
    miniarcs1=np.delete(miniarcs1,tuple(toerase))
    
    #print "\nDeleted elements on the other station: ",elim 
    for key in elim:
        miniarcs2[key]=np.delete(miniarcs2[key],tuple(elim[key]))
            
    #print "\nNew # of subarcs arc1: ",len(miniarcs1)#,miniarcs1
    #print "\nNew # of subarcs of arc2 (other station): ",len(miniarcs2)#,miniarcs2
            
    return miniarcs1,miniarcs2

def poly_fit(lI,time):
    #takes N elements from LI=L1-L2 and performs interpolation, 
    #detects datajumps in the diference between the polinomyal fit and real data 
    N=10 #window 
    tPoly ,Poly=[],[]
    
    for i in range(0,lI.size,N): 
        x=np.array(time[i:i+N])
        y=np.array(lI[i:i+N])
        z= np.polyfit(x,y,2)
        p = np.poly1d(z)
        for i in range(x.size):
            Poly.append(p(x[i]))
            tPoly.append(x[i]) 
    Poly=np.array(Poly)
    residual=lI-Poly
    jumps=datajump(residual,0.8)
    if jumps.size>0:
        pslip=np.argmax(residual[jumps])
        pslip=jumps[pslip]
    else:
        pslip=None
    return Poly,pslip


def outlier_detect(L,times,k=10):#k=30? 15 min
    outliers=[] #set of outlier factors for every element in L=L1-L2
    for i in range(0,L.size):
        if i<(k/2+1):
            neighbours=np.hstack((L[0:i],L[i+1:i+(k/2)+1])) #neighbours around i, without i
            tn=np.hstack((times[0:i],times[i+1:i+(k/2)+1]))
    
        elif i>L.size-(k/2+1):
            neighbours=np.hstack((L[i-k/2:i],L[i+1:L.size+1]))
            tn=np.hstack((times[i-k/2:i],times[i+1:L.size+1])) #times neighbour
            
        else:
            neighbours=np.hstack((L[i-k/2:i],L[i+1:i+(k/2)+1]))
            tn=np.hstack((times[i-k/2:i],times[i+1:i+(k/2)+1]))
        
        OFt=0
        deno=np.sum(1/(np.abs(times[i]-tn)*1.0))#denominador de Wpq para elemento i
        for neighbour in range(neighbours.size): 
            if times[neighbour]!=times[i]:
                Wpq=1/np.abs(times[i]-times[neighbour])
                Wpq=Wpq/deno
                OFt+=(Wpq*np.abs(L[i]-L[neighbour]))
        outliers.append(OFt) 
    
    outliers=np.array(outliers)
    if len(outliers)!=0:
        oslip=np.argmax(outliers) #term with biggest outlier factor
    else:
        oslip=None
    
    return outliers,oslip


def confirmed_slip(t,L):
    #if the biggest slip computed with polifit and outlier factor is the same an outlier is confirmed
    Poly,pslip=poly_fit(L,t)
    confirmed=[]
    while len(Poly)!=0: #if there are outliers
        __,oslip=outlier_detect(L,t*3600) 
        if pslip==oslip and pslip not in confirmed and pslip!=None: 
            confirmed.append(pslip) 
            L=np.delete(L,pslip) #remove outlier so it kepps looking for new different outliers
            t=np.delete(t,pslip)
            print "Poly: ",pslip,"Outlier factor: ",oslip
        Poly,npslip=poly_fit(L,t)  
        if npslip==pslip:
            break
        else:
            pslip=npslip   
    return confirmed
    
    
def remove_slip(miniarcs1,miniarcs2,oslip1): #deletes confirmed slip
    elim=miniarcs1[oslip1]
    print "Delete index: ",elim
    for subarc in range(len(miniarcs2)):
        if elim in miniarcs2[subarc]:
            i=np.where(miniarcs2[subarc]==elim)
            #print miniarcs2[subarc]
            miniarcs2[subarc]=np.delete(miniarcs2[subarc],i)
            #print miniarcs2[subarc]
    miniarcs1=np.delete(miniarcs1,oslip1)
    
    return miniarcs1,miniarcs2


def Smooth_code(R,L):
    size=R.size
    N=10
    #size/2
    
    for k in range(1,size):
        if k>N:
            n=N
        else:
            n=k
        #R[k]=L[k]+((n-1.0)/n)*(R[k-1]-L[k-1])+(1.0/n)*(R[k]-L[k])
        R[k]=1.0/n*R[k]+((n-1.0)/n)*(R[k-1]+(L[k]-L[k-1]))
     
    return R


def levelphase(ICODE,IPHASE,ELEV): 
    L=np.sum((ICODE-IPHASE)*(np.sin(ELEV)**2))/np.sum((np.sin(ELEV))**2) #leveling factor
    new_IPHASE=IPHASE+L
    return L,new_IPHASE
    


def datajumpMP(jumps1,jumps2, Code1L1,Code1L2,Phase1L1,Phase1L2,Code2L1,Code2L2,Phase2L1,Phase2L2):
    #Detect Multipass cycle slips
    #MP1 & MP2 In station 1 if MP slips are not in jumps, they are added
    gamma=gpstk.GAMMA_GPS
    MP11=Code1L1-(1+(2/(gamma-1)))*Phase1L1 + (2/(gamma-1))*Phase1L2
    MP12=Code1L2-((2*gamma*Phase1L1)/(gamma-1))+(((2*gamma)/(gamma-1))-1)*Phase1L2
    jumpsMP11=datajump(MP11,threshold=10)
    jumpsMP12=datajump(MP12,threshold=10)

    #Add multipass slips to cycle slips
    for i in jumpsMP11:
        if i not in jumps1:
            jumps1=np.append(jumps1,i)
            jumps1=np.sort(jumps1)

    for i in jumpsMP12:
        if i not in jumps1:
            jumps1=np.append(jumps1,i)
            jumps1=np.sort(jumps1)

    #MP1 & MP2 In station 2
    MP21=Code2L1-(1+(2/(gamma-1)))*Phase2L1 + (2/(gamma-1))*Phase2L2
    MP22=Code2L2-((2*gamma*Phase2L1)/(gamma-1))+(((2*gamma)/(gamma-1))-1)*Phase2L2
    jumpsMP21=datajump(MP21,threshold=10)
    jumpsMP22=datajump(MP22,threshold=10)


    for i in jumpsMP21:
        if i not in jumps2:
            jumps2=np.append(jumps2,i)
            jumps2=np.sort(jumps2)

    for i in jumpsMP22:
        if i not in jumps2:
            jumps2=np.append(jumps2,i)
            jumps2=np.sort(jumps2)

    return jumps1,jumps2













