import mongoose, {Schema} from 'mongoose';

const commonUserSchema = new Schema({
    user_id: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
    location: {
        type: String,
        required: true,
    },
    cases : [
        {
            case_id: {
                type: Schema.Types.ObjectId,
                ref: 'Case',
                required: true,
            },
            lawyer_id: {
                type: Schema.Types.ObjectId,
                ref: 'LawyerProfile',
                required: true,
            },
            status: {
                type: String,
                enum: ['ongoing', 'completed', 'closed'],
                default: 'ongoing',
            },
            last_updated : {
                type: Date,
                default: Date.now, 
            },  

        }
    ],
    last_login: {
        type: Date,
        default: Date.now,
    },
});

const CommonUser = mongoose.model('CommonUser', commonUserSchema);

export default CommonUser;