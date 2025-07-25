package xiaozhi.modules.sys.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.io.Serializable;

/**
 * 找回密码DTO
 */
@Data
@Schema(description = "retrieve password")
public class RetrievePasswordDTO implements Serializable {

    @Schema(description = "phone number")
    @NotBlank(message = "{sysuser.password.require}")
    private String phone;

    @Schema(description = "verification code")
    @NotBlank(message = "{sysuser.password.require}")
    private String code;

    @Schema(description = "new password")
    @NotBlank(message = "{sysuser.password.require}")
    private String password;



}