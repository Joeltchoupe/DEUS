class CRPCConvertParam:
    def __init__(self, methodName, paramIdx, paramName):
        self.methodName = methodName
        self.paramIdx = paramIdx
        self.paramName = paramName
      vRPCConvertParams = [
        CRPCConvertParam("sendmany", 1, "amounts"),
CRPCConvertParam("sendmany", 3, "minconf"),
CRPCConvertParam("addmultisigaddress", 0, "nrequired"),
CRPCConvertParam("addmultisigaddress", 1, "keys"),
CRPCConvertParam("createmultisig", 0, "nrequired"),
CRPCConvertParam("createmultisig", 1, "keys"),
CRPCConvertParam("listunspent", 0, "minconf"),
CRPCConvertParam("listunspent", 1, "maxconf"),
CRPCConvertParam("listunspent", 2, "as_of"),
CRPCConvertParam("listunspent", 3, "include_unconfirmed"),
CRPCConvertParam("listunspent", 4, "mindepth"),
CRPCConvertParam("listunspent", 5, "maxdepth"),
CRPCConvertParam("fundrawtransaction", 1, "fee_rate"),
CRPCConvertParam("fundrawtransaction", 3, "locktime"),
CRPCConvertParam("fundrawtransaction", 4, "replaceable"),
CRPCConvertParam("sendrawtransaction", 0, "hex"),
CRPCConvertParam("sendrawtransaction", 1, "allowhighfees"),
CRPCConvertParam("sendrawtransaction", 2, "maxfeerate"),
CRPCConvertParam("sendrawtransaction", 3, "minfeerate"),
CRPCConvertParam("gettransaction", 0, "txid"),
CRPCConvertParam("gettransaction", 1, "include_watchonly"),
CRPCConvertParam("getrawtransaction", 1, "verbose"),
CRPCConvertParam("decoderawtransaction", 1, "iswitness"),
CRPCConvertParam("gettransactiondetails", 0, "txid"),
CRPCConvertParam("gettransactiondetails", 1, "include_watchonly"),
CRPCConvertParam("listtransactions", 0, "account"),
CRPCConvertParam("listtransactions", 1, "count"),
CRPCConvertParam("listtransactions", 2, "skip"),
CRPCConvertParam("listtransactions", 3, "include_watchonly"),
CRPCConvertParam("listtransactions", 5, "category"),
CRPCConvertParam("listtransactions", 6, "blockhash"),
CRPCConvertParam("listtransactions", 7, "blockindex"),
CRPCConvertParam("listtransactions", 8, "timestart"),
CRPCConvertParam("listtransactions", 9, "timeend"),
CRPCConvertParam("listtransactions", 10, "include_removed"),
CRPCConvertParam("listtransactions", 11, "verbose"),
CRPCConvertParam("listsinceblock", 0, "blockhash"),
CRPCConvertParam("listsinceblock", 3, "include_watchonly"),
CRPCConvertParam("listsinceblock", 4, "include_removed"),
CRPCConvertParam("listsinceblock", 5, "verbose"),
CRPCConvertParam("gettransactionreceipt", 0, "txid"),
CRPCConvertParam("gettransactionreceipt", 1, "include_watchonly"),
CRPCConvertParam("getmempoolinfo", 0, "verbose"),
CRPCConvertParam("getblockhash", 0, "blockindex"),
CRPCConvertParam("getblockheader", 0, "hash"),
CRPCConvertParam("getblockheader", 1, "verbose"),
CRPCConvertParam("getblock", 0, "hash"),
CRPCConvertParam("getblock", 1, "verbose"),
CRPCConvertParam("getrawblock", 0, "hash"),
CRPCConvertParam("getrawblock", 1, "verbose"),
CRPCConvertParam("getblockstats", 0, "hash"),
CRPCConvertParam("getblockstats", 1, "stats"),
CRPCConvertParam("pruneblockchain", 0, "height"),
CRPCConvertParam("gettxout", 0, "txid"),
CRPCConvertParam("gettxout", 1, "n"),
CRPCConvertParam("gettxout", 2, "include_mempool"),
CRPCConvertParam("gettxoutproof", 0, "txids"),
CRPCConvertParam("gettxoutproof", 1, "blockhash"),
CRPCConvertParam("verifytxoutproof", 0, "proof"),
CRPCConvertParam("getspentinfo", 0, "txid")
rpc_params = [
    ("getrawmempool", 0, "verbose"),
    ("getrawmempool", 1, "mempool_sequence"),
    ("estimatesmartfee", 0, "conf_target"),
    ("estimaterawfee", 0, "conf_target"),
    ("estimaterawfee", 1, "threshold"),
    ("prioritisetransaction", 1, "dummy"),
    ("prioritisetransaction", 2, "fee_delta"),
    ("setban", 2, "bantime"),
    ("setban", 3, "absolute"),
    ("setnetworkactive", 0, "state"),
    ("setwalletflag", 1, "value"),
    ("getmempoolancestors", 1, "verbose"),
    ("getmempooldescendants", 1, "verbose"),
    ("gettxspendingprevout", 0, "outputs"),
    ("bumpfee", 1, "options"),
    ("bumpfee", 1, "conf_target"),
    ("bumpfee", 1, "fee_rate"),
    ("bumpfee", 1, "replaceable"),
    ("bumpfee", 1, "outputs"),
    ("bumpfee", 1, "original_change_index"),
    ("psbtbumpfee", 1, "options"),
    ("psbtbumpfee", 1, "conf_target"),
    ("psbtbumpfee", 1, "fee_rate"),
    ("psbtbumpfee", 1, "replaceable"),
    ("psbtbumpfee", 1, "outputs"),
    ("psbtbumpfee", 1, "original_change_index"),
    ("logging", 0, "include"),
    ("logging", 1, "exclude"),
    ("disconnectnode", 1, "nodeid"),
    ("upgradewallet", 0, "version"),
    # Echo with conversion (For testing only)
    ("echojson", 0, "arg0"),
    ("echojson", 1, "arg1"),
    ("echojson", 2, "arg2"),
    ("echojson", 3, "arg3"),
    ("echojson", 4, "arg4"),
    ("echojson", 5, "arg5"),
    ("echojson", 6, "arg6"),
    ("echojson", 7, "arg7"),
    ("echojson", 8, "arg8"),
    ("echojson", 9, "arg9"),
    ("rescanblockchain", 0, "start_height"),
    ("rescanblockchain", 1, "stop_height"),
    ("createwallet", 1, "disable_private_keys"),
    ("createwallet", 2, "blank"),
    ("createwallet", 4, "avoid_reuse"),
    ("createwallet", 5, "descriptors"),
    ("createwallet", 6, "load_on_startup"),
    ("createwallet", 7, "external_signer"),
    ("restorewallet", 2, "load_on_startup"),
    ("loadwallet", 1, "load_on_startup"),
    ("unloadwallet", 1, "load_on_startup"),
    ("getnodeaddresses", 0, "count"),
    ("addpeeraddress", 1, "port"),
    ("addpeeraddress", 2, "tried"),
    ("sendmsgtopeer", 0, "peer_id"),
    ("stop", 0, "wait"),
    ("addnode", 2, "v2transport"),
]

for cp in vRPCConvertParams:
            self.members.add((cp.methodName, cp.paramIdx))
            self.members_by_name.add((cp.methodName, cp.paramName))

    def arg_to_univalue(self, arg_value, method, param_idx):
        if (method, param_idx) in self.members:
            return json.loads(arg_value)
        else:
            return arg_value

    def arg_to_univalue_by_name(self, arg_value, method, param_name):
        if (method, param_name) in self.members_by_name:
            return json.loads(arg_value)
        else:
            return arg_value

rpc_cvt_table = CRPCConvertTable()

def rpc_convert_values(str_method, str_params):
    params = []

    for idx, str_param in enumerate(str_params):
        param = rpc_cvt_table.arg_to_univalue(str_param, str_method, idx)
        params.append(param)

    return params

def rpc_convert_named_values(str_method, str_params):
    params = {}
    positional_args = []

    for str_param in str_params:
        idx = str_param.find('=')
        if idx == -1:
            positional_args.append(rpc_cvt_table.arg_to_univalue(str_param, str_method, len(positional_args)))
        else:
            name = str_param[:idx]
            value = str_param[idx + 1:]

            # Overwrite earlier named values with later ones
            params[name] = rpc_cvt_table.arg_to_univalue_by_name(value, str_method, name)

    if positional_args:
        # Add positional args explicitly to avoid overwriting an explicit "args" value with an implicit one
        params['args'] = positional_args

    return params
