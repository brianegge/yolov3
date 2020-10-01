import base64
import json
import os
import ssl
import sys
try:
    import httplib # Python 2
except:
    import http.client as httplib # Python 3


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


def enrich(image_bytes, save_json):
    image_data = base64.b64encode(image_bytes).decode()
    
    headers = {"Content-type": "application/json",
               "X-Access-Token": "KhKhaWgY7Oku4p8TwYjW4bytJtzNCvyNfMPd"}
    params = json.dumps({"image": image_data})
    try:
        conn = httplib.HTTPSConnection("dev.sighthoundapi.com",
                context=ssl.SSLContext(ssl.PROTOCOL_TLSv1_2))
        conn.request("POST", "/v1/recognition?objectType=vehicle,licenseplate", params, headers)
        response = conn.getresponse()
    except:
        print("retrying sightound")
        conn = httplib.HTTPSConnection("dev.sighthoundapi.com",
                context=ssl.SSLContext(ssl.PROTOCOL_TLSv1_2))
        conn.request("POST", "/v1/recognition?objectType=vehicle,licenseplate", params, headers)
        response = conn.getresponse()

    result = json.loads(response.read())
    print("Detection Results = " + json.dumps(result, indent = 4 ))
    with open(save_json, 'w') as f:
        f.write(json.dumps(result, indent = 4))
    message = ""
    plates = []
    for o in result['objects']:
        annotations = o['vehicleAnnotation']
        if annotations['recognitionConfidence'] > 0.0:
            if 'attributes' in annotations:
                system = annotations['attributes']['system']
                if 'color' in system:
                    color = system['color']['name']
                    color = color.split('/')[0]
                    message += color + " "
                if 'make' in system and 'model' in system and system['make']['confidence'] > 0.6:
                    message += system['make']['name'] + " " + system['model']['name'] + " "
                elif 'vehicleType' in system:
                    message += system['vehicleType'] + " "
        if 'licenseplate' in annotations:
            plate = annotations['licenseplate']['attributes']['system']['string']
            if annotations['licenseplate']['attributes']['system']['region']['confidence'] > 0.55:
                region = annotations['licenseplate']['attributes']['system']['region']['name']
                plate['region'] = region
                if region in us_state_abbrev:
                    # if it's not one of the 50 states it's probably an error
                    plate['state'] = us_state_abbrev[region]
            plates.append(plate)
    return {'message':message,'plates':plates}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        f = "capture/vehicles/124743-shed-person.jpg"
    image_data = base64.b64encode(open(f, "rb").read()).decode()
    
    params = json.dumps({"image": image_data})
    conn.request("POST", "/v1/recognition?objectType=vehicle,licenseplate", params, headers)
    response = conn.getresponse()
    result = json.loads(response.read())
    print("Detection Results = " + json.dumps(result, indent = 4 ))
