import streamlit as st
import calendar
import requests

st.set_page_config(
    page_title="Hotel Reservation Prediction",
    page_icon="🚀",
    layout="centered",  
    initial_sidebar_state="expanded"
)

st.title("📋 Hotel Reservation Prediction")

with st.form("user_form"):
    st.subheader("Enter client details")
    lead_time = st.number_input(
        label="Lead Time",
        min_value=0,
        max_value=400,
        value=0,
        step=1
    )
    no_of_special_requests = st.number_input(
        label="Number of Special Requests",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )
    avg_price_per_room = st.number_input(
        label="Avg price per room",
        min_value=0.0,
        max_value=350.0,
        value=0.0,
        step=0.1,
        format="%.2f"
    )
    months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
    ]
    arrival_month = st.selectbox("Select a Arrival month", months)
    arrival_month_number = list(calendar.month_name).index(arrival_month)

    arrival_date = st.selectbox("Select Arrival date ", list(range(1, 32)))
    segement_type = {
        "Aviation":0,
        "Complementary":1,
        "Corporate":2,
        "Offline":3,
        "Online":4
    }
    market_segment_type = st.selectbox(
        "Select a Market Segment Type",
        segement_type.keys()
    )
    market_segment_type_num = segement_type[market_segment_type]

    no_of_week_nights = st.number_input(
        label="Number of Week Nights",
        min_value=0,
        max_value=6,
        value=0,
        step=1
    )
    no_of_weekend_nights = st.number_input(
        label="Number of Weekend Nights",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )
    meal_plan = {
        "Meal Plan 1":0,
        "Meal Plan 2":1,
        "Meal Plan 3":2,
        "Not Selected":3
    }
    type_of_meal_plan = st.selectbox(
        "Select a Meal Plan Type",
        meal_plan.keys()
    )
    type_of_meal_plan_num = meal_plan[type_of_meal_plan]

    room_type = {
        'Room_Type 1':0,
        'Room_Type 2':1,
        'Room_Type 3':2,
        'Room_Type 4':3,
        'Room_Type 5':4,
        'Room_Type 6':5,
        'Room_Type 7':6
    }
    room_type_reserved = st.selectbox(
        "Select a Room Type",
        room_type.keys()
    )
    room_type_reserved_num = room_type[room_type_reserved]


    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("Form submitted successfully!")
    st.write("### 👤 Cleint Details:")
    st.write(f"**Lead Time:** {lead_time}")
    st.write(f"**Number of Special Requests:** {no_of_special_requests}")
    st.write(f"**Avg price per room:** {avg_price_per_room}")
    st.write(f"**Arrival Month:** {arrival_month_number}")
    st.write(f"**Arrival Date:** {arrival_date}")
    st.write(f"**Market Segment Type:** {market_segment_type_num}")
    st.write(f"**Number of Week Nights:** {no_of_week_nights}")
    st.write(f"**Number of Weekend Nights:** {no_of_weekend_nights}")
    st.write(f"**Meal Plan Type:** {type_of_meal_plan_num}")
    st.write(f"**Room Type:** {room_type_reserved_num}")
    response = requests.post(f"http://localhost:8000/predict", json={"lead_time":lead_time,"no_of_special_requests":no_of_special_requests,"avg_price_per_room":avg_price_per_room,"arrival_month":arrival_month_number,"arrival_date":arrival_date,"market_segment_type":market_segment_type_num,"no_of_week_nights":no_of_week_nights,"no_of_weekend_nights":no_of_weekend_nights,"type_of_meal_plan":type_of_meal_plan_num,"room_type_reserved":room_type_reserved_num})
    if response.status_code == 200:
        if response.json()["output"] == 1:
            st.success(f"Client will not cancel his/her booking")
        else:
            st.warning(f"Client will cancel his/her booking")
    else:
        st.error("Failed to get response from api")

