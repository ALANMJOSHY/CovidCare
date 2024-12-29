from datetime import date


from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from.models import *
import joblib
# Create your views here.
def home(request):
    try:
        e=request.session['userid']
        if e:
            return render(request, 'home.html',{"e":e})
        else:
            pass
    except:
        return render(request,'home.html')

def Registration(request):
    return render(request,'Registration.html')
def save_registration(request):
    obj=Registration_tbl()
    obj.Name=request.POST.get("Name")
    obj.Location = request.POST.get("Location")
    obj.Email= request.POST.get("Email")
    obj.Password= request.POST.get("Password")
    obj.Mobile = request.POST.get("Mobile")
    obj.save()
    messages.success(request, 'Your message goes here.')
    return redirect('/Registration/')


def check_login(request):
    if request.method=="POST":
        email=request.POST.get("Email")
        password = request.POST.get("Password")
        if Registration_tbl.objects.filter(Email=email,Password=password).exists():
            us=Registration_tbl.objects.get(Email=email,Password=password)
            request.session['userid']=us.id
            return redirect("/user_home/")
        else:
            return redirect("/Registration/")
def Logout(request):
    del request.session['userid']
    return redirect('/')

def prediction(request):
    return render(request,'prediction.html')

def Adminlogin(request):
    return render(request,'Adminlogin.html')

def Adminlogincheck(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("/Adminhomepage/")
        else:
            return redirect("/Adminlogin/")

def Adminhomepage (request):
    return render(request, "Adminhomepage.html")

def check_prediction(request):
    age=int(request.POST.get("age"))
    gender=int(request.POST.get("gender"))
    if gender==1:
        gen="Male"
    elif gender==2:
        gen="Female"
    else:
        gen="Other"
    bmi=float(request.POST.get("bmi"))
    hypertensive=int(request.POST.get("hypertensive"))
    if hypertensive == 1:
        hyper="Yes"
    else:
        hyper="No"
    atrialfibrillation=int(request.POST.get("atrialfibrillation"))
    if atrialfibrillation == 1:
        atria="Yes"
    else:
        atria="No"
    chd=int(request.POST.get("chd"))
    if chd == 1:
        chd_data="Yes"
    else:
        chd_data="No"
    diabetes=int(request.POST.get("diabetes"))
    if diabetes==1:
        diabet="Yes"
    else:
        diabet="No"
    deficiencyanemias=int(request.POST.get("deficiencyanemias"))
    if deficiencyanemias==1:
        defi="Yes"
    else:
        defi="No"
    depression=int(request.POST.get("depression"))
    if depression==1:
        depress="Yes"
    else:
        depress="No"
    hyperlipemia=int(request.POST.get("hyperlipemia"))
    if hyperlipemia==1:
        hyperlip="Yes"
    else:
        hyperlip="No"

    renalfailure=int(request.POST.get("renalfailure"))
    if renalfailure==1:
        renal="Yes"
    else:
        renal="No"
    copd=int(request.POST.get("copd"))
    if copd==1:
        copd_data="Yes"
    else:
        copd_data="No"
    heartrate=float(request.POST.get("heartrate"))
    systolicbp=float(request.POST.get("systolicbp"))
    diastolicbp=float(request.POST.get("diastolicbp"))
    respiratoryrate=float(request.POST.get("respiratoryrate"))
    temperature=float(request.POST.get("temperature"))
    spo2=float(request.POST.get("spo2"))
    urineoutput=float(request.POST.get("urineoutput"))
    hematocrit=float(request.POST.get("hematocrit"))
    rbc=float(request.POST.get("rbc"))
    mch=float(request.POST.get("mch"))
    mchc=float(request.POST.get("mchc"))
    mcv=float(request.POST.get("mcv"))
    rdw=float(request.POST.get("rdw"))
    leucocyte=float(request.POST.get("leucocyte"))
    platelets=float(request.POST.get("platelets"))
    neutrophils=float(request.POST.get("neutrophils"))
    basophils=float(request.POST.get("basophils"))
    lymphocyte=float(request.POST.get("lymphocyte"))
    pt=float(request.POST.get("pt"))
    inr=float(request.POST.get("inr"))
    ntproBNP=float(request.POST.get("ntproBNP"))
    creatinekinase=float(request.POST.get("creatinekinase"))
    creatinine=float(request.POST.get("creatinine"))
    ureanitrogen=float(request.POST.get("ureanitrogen"))
    glucose=float(request.POST.get("glucose"))
    bloodpotassium=float(request.POST.get("bloodpotassium"))
    bloodsodium=float(request.POST.get("bloodsodium"))
    bloodcalcium=float(request.POST.get("bloodcalcium"))
    chloride=float(request.POST.get("chloride"))
    aniongap=float(request.POST.get("aniongap"))
    magnesiumion=float(request.POST.get("magnesiumion"))
    ph=float(request.POST.get("ph"))
    bicarbonate=float(request.POST.get("bicarbonate"))
    lacticacid=float(request.POST.get("lacticacid"))
    pco2=float(request.POST.get("pco2"))
    ef=float(request.POST.get("ef"))
    model=joblib.load("Covid_Mortality_model.pkl")
    n=[[age,gender,bmi,hypertensive,atrialfibrillation,chd,diabetes,deficiencyanemias,depression,hyperlipemia,renalfailure,copd,heartrate,
        systolicbp,diastolicbp,respiratoryrate,temperature,spo2,urineoutput,hematocrit,rbc,mch,mchc,mcv,rdw,leucocyte,platelets,
        neutrophils,basophils,lymphocyte,pt,inr,ntproBNP,creatinekinase,creatinine,ureanitrogen,glucose,bloodpotassium,bloodsodium,bloodcalcium,
        chloride,aniongap,magnesiumion,ph,
        bicarbonate,lacticacid,pco2,ef,2]]
    r=model.predict(n)
    print(r)
    if r == [0]:
        res="Based on your Symptoms the chance of mortality rate is low"
    else:
        res="Based on your input data the chance of mortality rate is high"
    obj=tbl_MedicalInformation()
    obj.age=age
    obj.gender=gen
    obj.chd=chd_data
    obj.bmi=bmi
    obj.hypertensive=hyper
    obj.atrialfibrillation=atria
    obj.diabetes=diabet
    obj.deficiencyanemias=defi
    obj.depression=depress
    obj.hyperlipemia=hyperlip
    obj.renalfailure=renal
    obj.copd=copd_data
    obj.heartrate=heartrate
    obj.systolicbp=systolicbp
    obj.diastolicbp=diastolicbp
    obj.respiratoryrate=respiratoryrate
    obj.temperature=temperature
    obj.spo2=spo2
    obj.urineoutput=urineoutput
    obj.hematocrit=hematocrit
    obj.rbc=rbc
    obj.mch=mch
    obj.mchc=mchc
    obj.mcv=mcv
    obj.rdw=rdw
    obj.leucocyte=leucocyte
    obj.platelets=platelets
    obj.neutrophils=neutrophils
    obj.basophils=basophils
    obj.lymphocyte=lymphocyte
    obj.pt=pt
    obj.inr=inr
    obj.ntproBNP=ntproBNP
    obj.creatinekinase=creatinekinase
    obj.creatinine=creatinine
    obj.ureanitrogen=ureanitrogen
    obj.glucose=glucose
    obj.bloodsodium=bloodsodium
    obj.bloodpotassium=bloodpotassium
    obj.bloodcalcium=bloodcalcium
    obj.chloride=chloride
    obj.aniongap=aniongap
    obj.magnesiumion=magnesiumion
    obj.ph=ph
    obj.bicarbonate=bicarbonate
    obj.lacticacid=lacticacid
    obj.pco2=pco2
    obj.ef=ef
    obj.user_id=request.session['userid']
    obj.result=res
    obj.save()
    return redirect("result",res=res)


def result(request,res):
    return render(request,"result.html",{"res":res})

def users(request):
    u=Registration_tbl.objects.all()
    return render(request,"users.html",{"u":u})

def delete_user(request,id):
    d=Registration_tbl.objects.get(id=id)
    d.delete()
    return redirect("/users/")


def predictions(request):
    p=tbl_MedicalInformation.objects.all()
    return render(request,"predictions.html",{"p":p})

def delete_predictions(request,id):
    d=tbl_MedicalInformation.objects.get(id=id)
    d.delete()
    return redirect("/predictions/")

def covid_info(request):
    return render(request, "covid_info.html")

def news(request):
    return render(request, "news.html")

def doctors(request):
    return render(request, "doctors.html")

def protect(request):
    return render(request, "protect.html")

def about(request):
    return render(request, "about.html")

def checkcovid19(request):
    return render(request,'checkcovid19.html')


def check_covid(request):
    breathing_problem = int(request.POST.get("breathingProblem"))
    if breathing_problem == 1:
        breathing_prob = "Yes"
    else:
        breathing_prob = "No"

    fever = int(request.POST.get("fever"))
    if fever == 1:
        fever_val = "Yes"
    else:
        fever_val = "No"

    dry_cough = int(request.POST.get("dryCough"))
    if dry_cough == 1:
        dry_cough_val = "Yes"
    else:
        dry_cough_val = "No"

    sore_throat = int(request.POST.get("soreThroat"))
    if sore_throat == 1:
        sore_throat_val = "Yes"
    else:
        sore_throat_val = "No"

    running_nose = int(request.POST.get("runningNose"))
    if running_nose == 1:
        running_nose_val = "Yes"
    else:
        running_nose_val = "No"

    asthma = int(request.POST.get("asthma"))
    if asthma == 1:
        asthma_val = "Yes"
    else:
        asthma_val = "No"

    chronic_lung_disease = int(request.POST.get("chronicLungDisease"))
    if chronic_lung_disease == 1:
        lung_disease_val = "Yes"
    else:
        lung_disease_val = "No"

    headache = int(request.POST.get("headache"))
    if headache == 1:
        headache_val = "Yes"
    else:
        headache_val = "No"

    heart_disease = int(request.POST.get("heartDisease"))
    if heart_disease == 1:
        heart_disease_val = "Yes"
    else:
        heart_disease_val = "No"

    diabetes = int(request.POST.get("diabetes"))
    if diabetes == 1:
        diabetes_val = "Yes"
    else:
        diabetes_val = "No"

    hyper_tension = int(request.POST.get("hyperTension"))
    if hyper_tension == 1:
        hyper_tension_val = "Yes"
    else:
        hyper_tension_val = "No"

    fatigue = int(request.POST.get("fatigue"))
    if fatigue == 1:
        fatigue_val = "Yes"
    else:
        fatigue_val = "No"

    gastrointestinal = int(request.POST.get("gastrointestinal"))
    if gastrointestinal == 1:
        gastrointestinal_val = "Yes"
    else:
        gastrointestinal_val = "No"

    abroad_travel = int(request.POST.get("abroadTravel"))
    if abroad_travel == 1:
        abroad_travel_val = "Yes"
    else:
        abroad_travel_val = "No"

    contact_with_covid_patient = int(request.POST.get("contactWithCovidPatient"))
    if contact_with_covid_patient == 1:
        contact_covid_patient_val = "Yes"
    else:
        contact_covid_patient_val = "No"

    attended_large_gathering = int(request.POST.get("attendedLargeGathering"))
    if attended_large_gathering == 1:
        large_gathering_val = "Yes"
    else:
        large_gathering_val = "No"

    visited_public_exposed_places = int(request.POST.get("visitedPublicExposedPlaces"))
    if visited_public_exposed_places == 1:
        public_exposed_places_val = "Yes"
    else:
        public_exposed_places_val = "No"

    family_working_in_public_exposed_places = int(request.POST.get("familyWorkingInPublicExposedPlaces"))
    if family_working_in_public_exposed_places == 1:
        family_working_val = "Yes"
    else:
        family_working_val = "No"

    wearing_masks = int(request.POST.get("wearingMasks"))
    if wearing_masks == 1:
        wearing_masks_val = "Yes"
    else:
        wearing_masks_val = "No"

    sanitization_from_market = int(request.POST.get("sanitizationFromMarket"))
    if sanitization_from_market == 1:
        sanitization_market_val = "Yes"
    else:
        sanitization_market_val = "No"

    # Your prediction logic goes here
    model = joblib.load("covid_check.pkl")
    n = [[breathing_problem, fever, dry_cough, sore_throat, running_nose, asthma, chronic_lung_disease, headache,
          heart_disease, diabetes, hyper_tension, fatigue, gastrointestinal, abroad_travel, contact_with_covid_patient,
          attended_large_gathering, visited_public_exposed_places, family_working_in_public_exposed_places,
          wearing_masks, sanitization_from_market]]
    r = model.predict(n)

    if r == [0]:
        res = "Based on your symptoms, the chance of covid 19 is not detected."
    else:
        res = "Based on your input data, the chance of covid 19 is detected."
    d=tbl_Covid19Prediction()
    d.breathingProblem = breathing_prob
    d.fever = fever_val
    d.dryCough = dry_cough_val
    d.soreThroat = sore_throat_val
    d.runningNose = running_nose_val
    d.asthma = asthma_val
    d.chronicLungDisease = lung_disease_val
    d.headache = headache_val
    d.heartDisease = heart_disease_val
    d.diabetes = diabetes_val
    d.hyperTension = hyper_tension_val
    d.fatigue = fatigue_val
    d.gastrointestinal = gastrointestinal_val
    d.abroadTravel = abroad_travel_val
    d.contactWithCovidPatient = contact_covid_patient_val
    d.attendedLargeGathering = large_gathering_val
    d.visitedPublicExposedPlaces = public_exposed_places_val
    d.familyWorkingInPublicExposedPlaces = family_working_val
    d.wearingMasks = wearing_masks_val
    d.sanitizationFromMarket = sanitization_market_val
    d.result=res
    d.user_id=request.session['userid']
    d.save()
    return render(request, "covid_result.html",{'res':res})

def covidchart(request):
    return render(request,'covidchart.html')

def doctor_reg(request):
    return render(request,'doctor_reg.html')

def save_doctor(request):
    obj = DoctorRegistration()
    obj.name = request.POST.get("Name")
    obj.email = request.POST.get("Email")
    obj.username = request.POST.get("Username")
    obj.password = request.POST.get("Password")
    obj.department = request.POST.get("Department")
    obj.save()
    messages.success(request, 'Your message goes here.')
    return redirect('/doctor_reg/')



def doctor_check_login(request):
    if request.method=="POST":
        username=request.POST.get("username")
        password = request.POST.get("password")
        if DoctorRegistration.objects.filter(username=username,password=password).exists():
            us=DoctorRegistration.objects.get(username=username,password=password)
            request.session['userid']=us.id
            return redirect("/doctor_home/")
        else:
            return redirect("/doctor_reg/")



def doctor_user(request):
    du=DoctorRegistration.objects.all()
    return render(request,"doctor_user.html",{"du":du})

def doctor_booking(request):
    d1=DoctorRegistration.objects.all()
    return render(request,'doctor_booking.html',{"d1":d1})

def contact_us(request):
    return render(request,'contact_us.html')

def doctor_home(request):
    d=DoctorRegistration.objects.get(id=request.session['userid'])
    d1=tbl_DoctorBooking.objects.filter(Doctor=request.session['userid'])
    d2=tbl_DoctorBooking.objects.filter(Doctor=request.session['userid'],date=date.today())

    return render(request,'doctor-dashboard.html',{"d":d,"d1":d1,"d2":d2})
""
def health_dept(request):
    return render(request,'health_dept.html')

def Healthlogincheck(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("/health_dept/")
        else:
            return redirect("/health_login/")

def health_login(request):
    return render(request,'health_login.html')

def health_notification(request):
    d=tbl_MedicalInformation.objects.filter(result="Based on your input data the chance of mortality rate is high")
    return render(request,'health_notification.html',{"d":d})

def search_hospital(request,location):
    d=Hospital.objects.filter(District__iexact=location)
    print(d)
    return render(request,'search_hospital.html',{"d":d})

def save_booking(request):
    if request.method == 'POST':
        obj = tbl_DoctorBooking()

        obj.full_name = request.POST.get("name")
        obj.phone_number = request.POST.get("phone")
        obj.email = request.POST.get("email")
        obj.date = request.POST.get("date")
        obj.time = request.POST.get("time")
        obj.area = request.POST.get("area")
        obj.city = request.POST.get("city")
        obj.state = request.POST.get("state")
        obj.post_code = request.POST.get("post-code")
        obj.Doctor_id=request.POST.get("doctor")
        obj.user_id=request.session['userid']
        obj.save()
        messages.success(request, 'Appointment booked successfully!')
        return redirect("/doctor_booking/")

def track_user(request,id):
    d=Registration_tbl.objects.get(id=id)
    mobile=d.mobile
    import phonenumbers
    from phonenumbers import geocoder
    # from test import number
    import folium

    Key = "6d6f969fd9024ac8afde957f0c86a5ba"

    number = "+91" + str(mobile)

    check_number = phonenumbers.parse(number)
    number_location = geocoder.description_for_number(check_number, "en")
    print(number_location)

    from phonenumbers import carrier
    service_provider = phonenumbers.parse(number)
    print(carrier.name_for_number(service_provider, "en"))
    service_provider1=carrier.name_for_number(service_provider, "en")
    from opencage.geocoder import OpenCageGeocode
    geocoder = OpenCageGeocode(Key)

    query = str(number_location)
    results = geocoder.geocode(query)

    lat = results[0]['geometry']['lat']
    lng = results[0]['geometry']['lng']
    print(lat, lng)
    import geopy
    from geopy.geocoders import Nominatim

    # Initialize the geolocator
    geolocator = Nominatim(user_agent="my_geocoder")

    # Define latitude and longitude
    latitude =lat
    longitude =lng

    # Get the address from latitude and longitude
    location = geolocator.reverse((latitude, longitude), language="en")

    # Print the address
    print(f"Address for latitude {latitude} and longitude {longitude}: {location.address}")
    address=f"Address for latitude {latitude} and longitude {longitude}: {location.address}"

    map_location = folium.Map(location=[lat, lng], zoom_start=9)
    folium.Marker([lat, lng], popup=number_location).add_to(map_location)
    map_location.save("mylocation.html")
    return render(request,"track_details.html",{"number_location":number_location,"service_provider1":service_provider1,"address":address})


def save_contactus(request):
    obj=contactus_tbl()
    obj.name=request.POST.get("Name")
    obj.email= request.POST.get("Email")
    obj.subject= request.POST.get("Subject")
    obj.save()
    return redirect('/')


def view_contact(request):
    u=contactus_tbl.objects.all()
    return render(request,"view_contact.html",{"u":u})


def delete_contact(request,id):
    d=contactus_tbl.objects.get(id=id)
    d.delete()
    return redirect("/view_contact/")

