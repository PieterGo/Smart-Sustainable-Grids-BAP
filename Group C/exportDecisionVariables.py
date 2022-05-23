import numpy
import pandas

def exportDecisionVariables(model, output_filename):

    T = len(model.T)
    B = len(model.B)
    P = len(model.P)
    E = len(model.E)
    G = len(model.G)
    X = len(model.X)

    powerOutput = numpy.zeros((I, T))
    CSU = numpy.zeros((I, T))
    CSD = numpy.zeros((I, T))
    u = numpy.zeros((I, T))

    essCharging = numpy.zeros((S, T))
    essDischarging = numpy.zeros((S, T))
    essSOE = numpy.zeros((S, T))
    u_ess = numpy.zeros((S, T))

    flow = numpy.zeros((L, T))
    voltage = numpy.zeros((N, T))

    pv = numpy.zeros((K, T))
    wind = numpy.zeros((Q, T))
    load = numpy.zeros((J, T))

    for t_in, t in enumerate(model.T):
        for i_in, i in enumerate(model.I):
            powerOutput[i_in, t_in] = model.P[i, t].value
            CSU[i_in, t_in] = model.CSU[i, t].value
            CSD[i_in, t_in] = model.CSD[i, t].value
            u[i_in, t_in] = model.u[i, t].value

        for s_in, s in enumerate(model.S):
            essCharging[s_in, t_in] = model.Pch[s, t].value
            essDischarging[s_in, t_in] = model.Pdis[s, t].value
            essSOE[s_in, t_in] = model.SOE[s, t].value
            u_ess[s_in, t_in] = model.u_ess[s, t].value

        for j_in, j in enumerate(model.J):
            load[j_in, t_in] = model.D[j, t].value

        for k_in, k in enumerate(model.K):
            pv[k_in, t_in] = model.PV[k, t].value
            
        for q_in, q in enumerate(model.Q):
            wind[q_in, t_in] = model.W[q, t].value

        for l_in, l in enumerate(model.L):
            flow[l_in, t_in] = model.Flow[l, t].value

        for n_in, n in enumerate(model.N):
            voltage[n_in, t_in] = model.theta[n, t].value

    powerOutputDF = pandas.DataFrame(index=model.I, columns=model.T, data=powerOutput)
    CSUDF = pandas.DataFrame(index=model.I, columns=model.T, data=CSU)
    CSDDF = pandas.DataFrame(index=model.I, columns=model.T, data=CSD)
    uDF = pandas.DataFrame(index=model.I, columns=model.T, data=u)
    essDischargingDF = pandas.DataFrame(index=model.S, columns=model.T, data=essDischarging)
    essChargingDF = pandas.DataFrame(index=model.S, columns=model.T, data=essCharging)
    essSOEDF = pandas.DataFrame(index=model.S, columns=model.T, data=essSOE)
    u_essDF = pandas.DataFrame(index=model.S, columns=model.T, data=u_ess)
    flowDF = pandas.DataFrame(index=model.L, columns=model.T, data=flow)
    voltageDF = pandas.DataFrame(index=model.N, columns=model.T, data=voltage)

    powerOutputDF.index.name = None
    powerOutputDF.columns.name = None
    CSUDF.index.name = None
    CSUDF.columns.name = None
    CSDDF.index.name = None
    CSDDF.columns.name = None
    uDF.index.name = None
    uDF.columns.name = None
    essDischargingDF.index.name = None
    essDischargingDF.columns.name = None
    essChargingDF.index.name = None
    essChargingDF.columns.name = None
    essSOEDF.index.name = None
    essSOEDF.columns.name = None
    u_essDF.index.name = None
    u_essDF.columns.name = None
    flowDF.index.name = None
    flowDF.columns.name = None
    voltageDF.index.name = None
    voltageDF.columns.name = None

    writer = pandas.ExcelWriter(output_filename)
    powerOutputDF.to_excel(writer, 'Generator_PowerOutput')
    CSUDF.to_excel(writer, 'Generator_StartUpCost')
    CSDDF.to_excel(writer, 'Generator_ShutDownCost')
    uDF.to_excel(writer, 'Generator_uDF')
    essChargingDF.to_excel(writer, 'ESSCharging')
    essDischargingDF.to_excel(writer, 'ESSDischarging')
    essSOEDF.to_excel(writer, 'ESSSOE')
    u_essDF.to_excel(writer, 'UESS')
    flowDF.to_excel(writer, 'Flow')
    voltageDF.to_excel(writer, 'Voltage_Angle')
    writer.save()

