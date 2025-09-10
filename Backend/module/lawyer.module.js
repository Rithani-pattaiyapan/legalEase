import mongoose , {Schema} from 'mongoose';
import { use } from 'react';

const lawyerSchema = new Schema({
    user_id: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
    specialization: {
        type: String,
        enum: ['criminal', 'civil', 'family', 'corporate', 'property', 'labor', 'tax', 'intellectual_property', 'environmental', 'human_rights', 'others'],
        required: true,
    },
    experience_years: {
        type: Number,
        required : true,
        min : 0,
    },
    location: {
        type: String,
        required : true,
    },
    availability: {
        type: Boolean,
        default: true,
    },
    ratings: {
        type: Number,
        default: 0,
        min : 0,
        max : 5
    },

    // cases
    cases: [
        {
            case_id: {
                type: Schema.Types.ObjectId,
                ref : 'Case',
                required: true,
            },
            client_id: {
                type : Schema.Types.ObjectId,
                ref: 'CommonUser',
                required: true,
            },
            status: {
                type: String,
                enum: ['ongoing','completed','closed'],
                default : 'ongoing'
            },
            last_update: {
                type: Date,
                default: Date.now,
            }
        }
    ],

    // verification

    bar_council_id:{
        type: String,
        required: true,
        trim : true,
    },
    bar_council_state:{
        type: String,
        required: true,
    },
    is_verified: {
        type: Boolean,
        default: false,
    },

    // Admin approval
    approved_status: {
        type: String,
        enum: ['pending', 'approved', 'rejected'],
        default: 'pending',
    }
});

const LawyerProfile = mongoose.model('LawyerProfile', lawyerSchema);

export default LawyerProfile;