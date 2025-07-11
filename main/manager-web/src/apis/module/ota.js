import { getServiceUrl } from '../api';
import RequestService from '../httpRequest';

export default {
    // 分页查询OTA固件信息
    getOtaList(params, callback) {
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag`)
            .method('GET')
            .data(params)
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to get OTA firmware list:', err);
                RequestService.reAjaxFun(() => {
                    this.getOtaList(params, callback);
                });
            }).send();
    },
    // 获取单个OTA固件信息
    getOtaInfo(id, callback) {
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag/${id}`)
            .method('GET')
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to get OTA firmware info:', err);
                RequestService.reAjaxFun(() => {
                    this.getOtaInfo(id, callback);
                });
            }).send();
    },
    // 保存OTA固件信息
    saveOta(entity, callback) {
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag`)
            .method('POST')
            .data(entity)
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to save OTA firmware info:', err);
                RequestService.reAjaxFun(() => {
                    this.saveOta(entity, callback);
                });
            }).send();
    },
    // 更新OTA固件信息
    updateOta(id, entity, callback) {
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag/${id}`)
            .method('PUT')
            .data(entity)
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to uodate OTA firmware info:', err);
                RequestService.reAjaxFun(() => {
                    this.updateOta(id, entity, callback);
                });
            }).send();
    },
    // 删除OTA固件
    deleteOta(id, callback) {
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag/${id}`)
            .method('DELETE')
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to delete OTA firmware:', err);
                RequestService.reAjaxFun(() => {
                    this.deleteOta(id, callback);
                });
            }).send();
    },
    // 上传固件文件
    uploadFirmware(file, callback) {
        const formData = new FormData();
        formData.append('file', file);
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag/upload`)
            .method('POST')
            .data(formData)
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to upload OTA firmware files:', err);
                RequestService.reAjaxFun(() => {
                    this.uploadFirmware(file, callback);
                });
            }).send();
    },
    // 获取固件下载链接
    getDownloadUrl(id, callback) {
        RequestService.sendRequest()
            .url(`${getServiceUrl()}/otaMag/getDownloadUrl/${id}`)
            .method('GET')
            .success((res) => {
                RequestService.clearRequestTime();
                callback(res);
            })
            .networkFail((err) => {
                console.error('failed to get download url:', err);
                RequestService.reAjaxFun(() => {
                    this.getDownloadUrl(id, callback);
                });
            }).send();
    }
}