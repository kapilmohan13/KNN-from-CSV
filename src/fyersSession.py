from fyers_apiv3 import fyersModel
import login
import credential as cr

# %%
def getFyerSession():
    # Create a session object to handle the Fyers API authentication and token generation
    session = fyersModel.SessionModel(
        client_id=cr.client_id,
        secret_key=cr.secret_key, 
        redirect_uri=cr.redirect_uri, 
        response_type=cr.response_type, 
        grant_type=cr.grant_type
    )

    # Set the authorization code in the session object
    session.set_token(login.getAuthCode())

    # Generate the access token using the authorization code
    response = session.generate_token()

    access_token=response['access_token']
    # refresh_token=response['refresh_token']

    # Initialize the FyersModel instance with your client_id, access_token, and enable async mode
    fyers = fyersModel.FyersModel(client_id=cr.client_id, is_async=False, token=access_token, log_path="")
    return fyers

