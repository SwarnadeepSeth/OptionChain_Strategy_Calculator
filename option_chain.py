import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

def call_payoff(PlotRange, strike_price, premium):
    return np.where(PlotRange > strike_price, PlotRange - strike_price, 0) - premium

def put_payoff(PlotRange, strike_price, premium):
    return np.where(PlotRange < strike_price, strike_price - PlotRange, 0) - premium

def PL_colorfill(x, y):
    # create a mask for positive and negative values
    pos_mask = y >= 0
    neg_mask = y < 0

    # create a trace for positive and negative values
    fig.add_trace(go.Scatter(
        x=x[pos_mask], y=y[pos_mask], fill='tozeroy', fillcolor='rgba(26,150,65,0.5)', line=dict(color='black', width=2), name='Profit', showlegend=False)
    )

    fig.add_trace(go.Scatter(
        x=x[neg_mask], y=y[neg_mask], fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(color='black', width=2), name='Loss', showlegend=False)
    )

def options_chain(tk):
    # Expiration dates
    exps = tk.options
    exps = exps[:5] # only the first 5 expirations
    print (exps)

    # Get options for each expiration
    options = pd.DataFrame()

    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.concat([opt.calls, opt.puts], axis=0)
        opt['expirationDate'] = e
        options = pd.concat([options, opt], ignore_index=True)
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = round(0.5*(options['bid'] + options['ask']), 2) # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSymbol', 'contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])
    options = options.rename(columns = {'strike': 'Strike Price', 'mark': 'Ask Price', 'volume': 'Volume', 'openInterest': 'Open Interest', 'impliedVolatility': 'Volatility'})
    options["Volatility"] = options["Volatility"].apply(lambda x: round(x*100, 2))
    return options

# ============================================================================= #
st.title("Option chain and Profit/Loss Calculator (With Customized Strategy)")

form = st.form(key='Stock Ticker')
ticker = form.text_input(label="**:blue[Enter the US Stock Ticker:]**", value="")
btn = form.form_submit_button(label='Submit')

if btn:
    tk = yf.Ticker(ticker)

    opt_chain = options_chain(tk)
    opt_chain["Quantity(Editable)"] = 1
    opt_chain = opt_chain.sort_values(by=['Strike Price'], ascending=False)
    ohlc_data = tk.history(period="1y")

    current_close = round(ohlc_data["Close"][-1], 2)
    pct_chg = round(100*(current_close - ohlc_data["Close"][-2])/ohlc_data["Close"][0], 2)

    price_show = ticker+" $"+str(current_close)+" ("+str(pct_chg)+"%)"
    st.header(price_show)

    # Get the expiration dates
    exps = opt_chain['expirationDate'].unique()
    # sort the expiration dates
    exps = sorted(exps, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

    st.write("--------------------------------------------------------------------------------------------------------------")
    option_date = st.selectbox("Select the Expiration Date:", (exps))
    st.write('You selected:', option_date)
    st.write("--------------------------------------------------------------------------------------------------------------")
    st.write("Option Chain:")

    # ============================================================================= #
    tab1, tab2, tab3, tab4 = st.tabs(["BUY CALL", "SELL CALL", "BUY PUT", "SELL PUT"])

    buy_call_chain = opt_chain[(opt_chain['expirationDate'] == option_date) & (opt_chain['CALL'] == True) & (abs(opt_chain['Strike Price']-current_close) <0.15*current_close)]
    buy_call_chain = buy_call_chain.drop(columns = ['bid', 'ask', 'expirationDate', 'CALL', 'inTheMoney'])

    sell_call_chain = buy_call_chain.rename(columns = {'Ask Price': 'Bid Price'})

    buy_put_chain = opt_chain[(opt_chain['expirationDate'] == option_date) & (opt_chain['CALL'] == False) & (abs(opt_chain['Strike Price']-current_close) <0.15*current_close)]
    buy_put_chain = buy_put_chain.drop(columns = ['bid', 'ask', 'expirationDate', 'CALL', 'inTheMoney'])

    sell_put_chain = buy_put_chain.rename(columns = {'Ask Price': 'Bid Price'})

    gd_tab1 = GridOptionsBuilder.from_dataframe(buy_call_chain)
    gd_tab2 = GridOptionsBuilder.from_dataframe(sell_call_chain)
    gd_tab3 = GridOptionsBuilder.from_dataframe(buy_put_chain)
    gd_tab4 = GridOptionsBuilder.from_dataframe(sell_put_chain)

    with tab1:
        gd_tab1.configure_column("Strike Price", cellStyle={'color': 'red', 'textAlign': 'center'})
        gd_tab1.configure_column("Ask Price", cellStyle={'color': 'blue'})
        gd_tab1.configure_selection(selection_mode='multiple', use_checkbox=True)
        gd_tab1.configure_column("Quantity(Editable)", editable=True)
        grid_table_tab1 = AgGrid(buy_call_chain, gridOptions=gd_tab1.build(), update_mode='MANUAL', key = 'tab1_grid')

        st.write('## Selected BUY CALLS')
        tab1_selected_row = grid_table_tab1["selected_rows"]
        st.dataframe(tab1_selected_row)


    with tab2:
        gd_tab2.configure_column("Strike Price", cellStyle={'color': 'red', 'textAlign': 'center'})
        gd_tab2.configure_column("Bid Price", cellStyle={'color': 'blue'})
        gd_tab2.configure_selection(selection_mode='multiple', use_checkbox=True)
        gd_tab2.configure_column("Quantity(Editable)", editable=True)
        grid_table_tab2 = AgGrid(sell_call_chain, gridOptions=gd_tab2.build(), update_mode='MANUAL', key = 'tab2_grid')

        st.write('## Selected SELL CALLS')
        tab2_selected_row = grid_table_tab2["selected_rows"]
        st.dataframe(tab2_selected_row)

    with tab3:
        gd_tab3.configure_column("Strike Price", cellStyle={'color': 'red', 'textAlign': 'center'})
        gd_tab3.configure_column("Ask Price", cellStyle={'color': 'blue'})
        gd_tab3.configure_selection(selection_mode='multiple', use_checkbox=True)
        gd_tab3.configure_column("Quantity(Editable)", editable=True)
        grid_table_tab3 = AgGrid(buy_put_chain, gridOptions=gd_tab3.build(), update_mode='MANUAL', key = 'tab3_grid')

        st.write('## Selected BUY PUTS')
        tab3_selected_row = grid_table_tab3["selected_rows"]
        st.dataframe(tab3_selected_row)

    with tab4:
        gd_tab4.configure_column("Strike Price", cellStyle={'color': 'red', 'textAlign': 'center'})
        gd_tab4.configure_column("Bid Price", cellStyle={'color': 'blue'})
        gd_tab4.configure_selection(selection_mode='multiple', use_checkbox=True)
        gd_tab4.configure_column("Quantity(Editable)", editable=True)
        grid_table_tab4 = AgGrid(sell_put_chain, gridOptions=gd_tab4.build(), update_mode='MANUAL', key = 'tab4_grid')

        st.write('## Selected SELL PUTS')
        tab4_selected_row = grid_table_tab4["selected_rows"]
        st.dataframe(tab4_selected_row)

    # ============================================================================= #
    spot_price = current_close
    PlotRange = np.arange(0.8*spot_price, 1.2*spot_price, 0.01)
    payoff = np.zeros(len(PlotRange))
    premium = 0

    for i in range (len(tab1_selected_row)):
        tab1_strike_price = tab1_selected_row[i]['Strike Price']
        tab1_premium = tab1_selected_row[i]['Ask Price']
        qnt1 = int(tab1_selected_row[i]['Quantity(Editable)'])

        payoff_long_call = call_payoff(PlotRange, tab1_strike_price, tab1_premium)
        payoff += qnt1*payoff_long_call
        premium += qnt1*tab1_premium

    for i in range (len(tab2_selected_row)):
        tab2_strike_price = tab2_selected_row[i]['Strike Price']
        tab2_premium = tab2_selected_row[i]['Bid Price']
        qnt2 = int(tab2_selected_row[i]['Quantity(Editable)'])

        payoff_short_call = -1.0*call_payoff(PlotRange, tab2_strike_price, tab2_premium)
        payoff += qnt2*payoff_short_call
        premium -= qnt2*tab2_premium

    for i in range (len(tab3_selected_row)):    
        tab3_strike_price = tab3_selected_row[i]['Strike Price']
        tab3_premium = tab3_selected_row[i]['Ask Price']
        qnt3 = int(tab3_selected_row[i]['Quantity(Editable)'])

        payoff_long_put = put_payoff(PlotRange, tab3_strike_price, tab3_premium)
        payoff += qnt3*payoff_long_put
        premium += qnt3*tab3_premium

    for i in range (len(tab4_selected_row)):        
        tab4_strike_price = tab4_selected_row[i]['Strike Price']
        tab4_premium = tab4_selected_row[i]['Bid Price']
        qnt4 = int(tab4_selected_row[i]['Quantity(Editable)'])

        payoff_short_put = -1.0*put_payoff(PlotRange, tab4_strike_price, tab4_premium)
        payoff += qnt4*payoff_short_put
        premium -= qnt4*tab4_premium

    payoff = 100*payoff
    # ============================================================================= #
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=PlotRange, y=payoff, mode='lines', name='Strategy', line_color='black'))
    PL_colorfill(PlotRange, payoff)

    st.write('<p style="color:green;font-size: 20px;">Maximum Profit [within the shown range]:</p>', f'<p style="color:green;font-size: 20px;">{str(round(payoff[np.where(payoff==max(payoff))][0], 2))}</p>', unsafe_allow_html=True)
    st.write('<p style="color:red;font-size: 20px;">Maximum Loss [within the shown range]:</p>', f'<p style="color:red;font-size: 20px;">{str(round(payoff[np.where(payoff==min(payoff))][0], 2))}</p>', unsafe_allow_html=True)
    premium_need_str = "Margin Needed: "+ str(round(100*premium, 2))
    st.write(f'<p style="color:black;font-size: 20px;">{premium_need_str}</p>', unsafe_allow_html=True)

    zero_crossings = np.where(np.diff(np.sign(payoff)))[0]

    if (len(zero_crossings) == 1):
        st.write("Breakeven Price:", round(PlotRange[zero_crossings[0]],2))

    if (len(zero_crossings) == 2):
        st.write("Lower Breakeven Price:", round(PlotRange[zero_crossings[0]],2))
        st.write("Upper Breakeven Price:", round(PlotRange[zero_crossings[1]],2))

    fig.add_vline(x=spot_price, line_width=1, line_color="blue")

    # format the layout
    fig.update_layout(
        plot_bgcolor="#FFF", 
        hovermode="x",
        hoverdistance=500, # Distance to show hover label of data point
        spikedistance=1000, # Distance to show spike
        xaxis=dict(
            title="Price",
            linecolor="#BCCCDC",  
            showgrid=False,
            titlefont=dict(size=20),
            tickfont = dict(size=18),
            showspikes=True, # Show spike line for X-axis
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across"
        ),
        yaxis=dict(
            title="Profit/Loss",
            linecolor="#BCCCDC", 
            zeroline=True,
            zerolinewidth=1, 
            zerolinecolor='black',
            showgrid=False, 
            titlefont=dict(size=20),
            tickfont = dict(size=18)
        ),
        legend=dict(
            x=0.45, y=0.98, 
            font=dict(size=15),
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant',
            itemclick=False,
            itemdoubleclick=False
        )
    )

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
