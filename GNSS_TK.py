import numpy as np
import gpstk
import matplotlib.pyplot as plt

def getdata(nfile,ofile,strsat=None): #one week data
    f1=gpstk.L1_FREQ_GPS
    f2=gpstk.L2_FREQ_GPS
    t=[] #observation epoch on code and phase 
    Iphase,Icode,IPPS={},{},{} #dicc: key=time observer; values=iono delay and IPP
    VTECphase,VTECcode={},{}
    ELEV={}
    oheader,odata=gpstk.readRinex3Obs(ofile,strict=True) 
    nheader,ndata=gpstk.readRinex3Nav(nfile)
    bcestore = gpstk.GPSEphemerisStore() 
    
    for ndato in ndata:
        ephem = ndato.toGPSEphemeris()
        bcestore.addEphemeris(ephem)
    bcestore.SearchNear() 
   
    
    alfa=f1**2/(f1**2-f2**2)
    for observation in odata:
        sats=[satID for satID, datumList in observation.obs.iteritems() if str(satID).split()[0]=="GPS" ] 
        obs_types = np.array([i for i in oheader.R2ObsTypes])
        if 'C1' and 'P2' and 'L1' and 'L2' in obs_types:
            for sat in sats:
                if  str(sat)==strsat :#Return for a specific satellite
                    eph = bcestore.findEphemeris(sat, observation.time) 
                    sat_pos = eph.svXvt(observation.time)
                    rec_pos = gpstk.Position(oheader.antennaPosition[0], oheader.antennaPosition[1], oheader.antennaPosition[2]).asECEF()
                    elev = oheader.antennaPosition.elvAngle(sat_pos.x)
                    azim = oheader.antennaPosition.azAngle(sat_pos.x)
                    time=observation.time
                    R=6.378e6 #earth radius
                    mapp=1/np.cos(np.arcsin(R/(R+350000))*np.sin(elev))
                    t.append(np.trunc(gpstk.YDSTime(time).sod))
                    IPP=rec_pos.getIonosphericPiercePoint(elev, azim, 350000).asECEF()
                   
                    if np.size(np.where(obs_types=='C1'))!=0 and np.size(np.where(obs_types=='P2'))!=0 and np.size(np.where(obs_types=='L1'))!=0 and np.size(np.where(obs_types=='L2'))!=0: 
                        C1_idx = np.where(obs_types=='C1')[0][0] 
                        P2_idx = np.where(obs_types=='P2')[0][0]
                        R1=observation.getObs(sat, C1_idx).data 
                        R2=observation.getObs(sat, P2_idx).data
                        L1_idx = np.where(obs_types=='L1')[0][0]
                        L2_idx = np.where(obs_types=='L2')[0][0]
                        L1=observation.getObs(sat, L1_idx).data*gpstk.L1_WAVELENGTH_GPS 
                        L2=observation.getObs(sat, L2_idx).data*gpstk.L2_WAVELENGTH_GPS
                       
                        if R2<1e8 and R1<1e8 and L2<1e8 and L1<1e8: #Distances should be in order of 1e7 meters, more than that is considered an error  
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
                            #stec=(iono_delay_p*f1**2)/(-40.3) #STEC delay on phase [mm]
                            #vtec=stec/mapp #vertical delay!
        else:
            print "Needs both L1 and L2 frequencies to compute delay"
            break
    
    return t,Icode,Iphase,VTECphase,ELEV,IPPS

def getdata_stationpair(station1,station2,strsat=None): 
    s1n_file,s1o_file=station1[0],station1[1] #station pair
    s2n_file,s2o_file=station2[0],station2[1] 
    
    t1,Icode1,Iphase1,VTECphase1,ELEV1,IPP1=getdata(s1n_file,s1o_file,strsat)
    t2,Icode2,Iphase2,VTECphase2,ELEV2,IPP2=getdata(s2n_file,s2o_file,strsat)
    
    return t1,t2,Icode1,Iphase1,Icode2,Iphase2,VTECphase1,VTECphase2,ELEV1,ELEV2,IPP1,IPP2

def get_obstimes(t,first,last):
    t=np.array(t)
    #new_t=t[(t>=first) & (t<=last)]
    new_t=t[np.logical_and(t>=first, t<=last)]
    #indices=np.where(np.logical_and(t>=first, t<=last))
    return new_t#,indices[0]

def datajump(lI,times,threshold=0.5): #lInput: lI=L1-L2, times.
    jumps=[]
    jumps = np.where(np.diff(np.hstack(([0],lI)))>threshold)
    return jumps[0]
    
def sub_arcs(lI,t,jumps):
    miniarcs=np.split(lI,jumps)
    ntimes=np.split(t,jumps)
    return miniarcs,ntimes
    
def remove_short(miniarcs,ntimes):
    i=0
    for arc in miniarcs:
        if arc.size<10:
            miniarcs=np.delete(miniarcs,i)
            ntimes=np.delete(ntimes,i)
        i+=1
    return miniarcs,ntimes
  
#takes N elements from LI=L1-L2 and performs interpolation, 
#detcts datajumps in the diference between the polinomyal fit and real data 
def poly_fit(lI,time):
    N=10 #window 
    tPoly=[]
    Poly=[]
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
    jumps=datajump(residual,time,0.8)
    if jumps.size>0:
        pslip=np.argmax(residual[jumps])
    else:
        pslip=0
    return Poly,pslip

def outlier_detect(L,times,k=10):
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
    oslip=np.argmax(outliers) #term with biggest outlier factor
    
    return outliers,oslip
    
#if the biggest slip computed with polifit and outlier factor is the same
def fixslip(t,L,threshold=0.8):
    Poly,pslip=poly_fit(L,t) #residuals bigger than threshold 0.8, and biggest slip detected
    confirmed=[]#confirmed outliers
    while len(Poly)!=0: #if there are outliers
        __,oslip=outlier_detect(L,t*3600) #biggest slip detected with outlier factor
        if pslip==oslip and pslip not in confirmed and pslip!=0:
            confirmed.append(pslip) #recorded as an outlier
            L=np.delete(L,pslip) #remove outlier
            t=np.delete(t,pslip)
            print "Poly: ",pslip,"Outlier factor: ",oslip
        Poly,npslip=poly_fit(L,t)  
        if npslip==pslip:
            break
        else:
            pslip=npslip
    return L,t,confirmed
    
def levelphase(ICODE,IPHASE,ELEV):
    #L=np.sum((ICODE-IPHASE)*(np.sin(ELEV)**2))/np.sum((np.sin(ELEV))**2) #leveling factor
    L=np.sum(ICODE-IPHASE)/ICODE.size #leveling factor
    new_IPHASE=IPHASE+L
    return L,new_IPHASE
    
